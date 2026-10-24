import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { GoogleGenAI, Modality } from '@google/genai';

const execFileAsync = promisify(execFile);

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error('Missing GEMINI_API_KEY');
  process.exit(1);
}

const ai = new GoogleGenAI({
  apiKey: API_KEY,
  httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
});

const langConfigs = {
  ar: {
    voice: 'Fenrir',
    dir: 'audio/ar',
    instruction: 'You are a passionate native Arabic football commentator in a recording booth. Read aloud ONLY the script line in Arabic with dramatic stadium passion (Issam Chaouali style). NEVER read stage directions or brackets out loud. Speak ONLY the Arabic line.',
  },
  fr: {
    voice: 'Kore',
    dir: 'audio/fr',
    instruction: 'You are a dynamic native French football commentator in a recording booth. Read aloud ONLY the script line in French with authentic French broadcast energy. NEVER read stage directions or brackets out loud. Speak ONLY the French line.',
  },
  es: {
    voice: 'Puck',
    dir: 'audio/es',
    instruction: 'You are an explosive native Spanish football commentator in a recording booth. Read aloud ONLY the script line in Spanish with high-energy commentator emotion (¡Gooool!). NEVER read stage directions or brackets out loud. Speak ONLY the Spanish line.',
  },
};

function cleanScriptLine(phrase) {
  const regex = /\[([^\]]+)\]\s*([^\[]*)/g;
  const spoken = [];
  const emotions = [];
  let match;
  while ((match = regex.exec(phrase)) !== null) {
    const emo = match[1].trim();
    let txt = match[2].trim();
    if (!txt) continue;
    emotions.push(emo);
    spoken.push(txt);
  }
  const line = spoken.length > 0 ? spoken.join(' ') : phrase.replace(/\[[^\]]+\]/g, '').trim();
  const emoDesc = emotions.length > 0 ? emotions.join(', ') : 'passionate';
  return { line, emoDesc };
}

async function savePcmAsWavAndMp3(pcmBuffer, baseOutPath) {
  const uniqueId = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const rawPath = `${baseOutPath}_${uniqueId}.pcm`;
  const wavPath = `${baseOutPath}.wav`;
  const mp3Path = `${baseOutPath}.mp3`;

  fs.writeFileSync(rawPath, pcmBuffer);
  try {
    await execFileAsync('ffmpeg', [
      '-y',
      '-f', 's16le',
      '-ar', '24000',
      '-ac', '1',
      '-i', rawPath,
      wavPath,
    ]);
    await execFileAsync('ffmpeg', [
      '-y',
      '-f', 's16le',
      '-ar', '24000',
      '-ac', '1',
      '-i', rawPath,
      '-codec:a', 'libmp3lame',
      '-qscale:a', '2',
      mp3Path,
    ]);
  } finally {
    if (fs.existsSync(rawPath)) fs.unlinkSync(rawPath);
  }
}

class LiveVoiceWorker {
  constructor(voiceName, langInstruction) {
    this.voiceName = voiceName;
    this.langInstruction = langInstruction;
    this.session = null;
    this.turnsUsed = 0;
    this.chunks = [];
    this.resolveTurn = null;
    this.rejectTurn = null;
  }

  async connect() {
    if (this.session) {
      try { this.session.close(); } catch {}
      this.session = null;
    }
    this.turnsUsed = 0;

    this.session = await ai.live.connect({
      model: 'gemini-3.8-live',
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: this.voiceName },
          },
        },
        systemInstruction: {
          parts: [{ text: this.langInstruction }],
        },
      },
      callbacks: {
        onmessage: (msg) => {
          const parts = msg.serverContent?.modelTurn?.parts || [];
          for (const p of parts) {
            if (p.inlineData?.data) {
              this.chunks.push(Buffer.from(p.inlineData.data, 'base64'));
            }
          }
          if (msg.serverContent?.turnComplete) {
            if (this.resolveTurn) {
              const fn = this.resolveTurn;
              this.resolveTurn = null;
              this.rejectTurn = null;
              fn(Buffer.concat(this.chunks));
            }
          }
        },
        onerror: (err) => {
          if (this.rejectTurn) {
            const fn = this.rejectTurn;
            this.resolveTurn = null;
            this.rejectTurn = null;
            fn(err);
          }
        },
        onclose: () => {
          if (this.rejectTurn) {
            const fn = this.rejectTurn;
            this.resolveTurn = null;
            this.rejectTurn = null;
            fn(new Error('Session closed'));
          }
          this.session = null;
        },
      },
    });
  }

  async synthesize(prompt, attempt = 1) {
    try {
      if (!this.session || this.turnsUsed >= 6) {
        await this.connect();
      }
      this.chunks = [];
      this.turnsUsed++;

      const pcmBuffer = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          this.resolveTurn = null;
          this.rejectTurn = null;
          reject(new Error('Turn timeout after 25s'));
        }, 25000);

        this.resolveTurn = (buf) => {
          clearTimeout(timer);
          resolve(buf);
        };
        this.rejectTurn = (err) => {
          clearTimeout(timer);
          reject(err);
        };

        this.session.sendClientContent({
          turns: [{ role: 'user', parts: [{ text: prompt }] }],
          turnComplete: true,
        });
      });

      if (!pcmBuffer || pcmBuffer.length < 3000) {
        throw new Error(`Audio too short (${pcmBuffer ? pcmBuffer.length : 0} bytes)`);
      }
      return pcmBuffer;
    } catch (err) {
      this.session = null;
      if (attempt < 4) {
        await new Promise((r) => setTimeout(r, 800 * attempt));
        return this.synthesize(prompt, attempt + 1);
      }
      throw err;
    }
  }

  close() {
    if (this.session) {
      try { this.session.close(); } catch {}
      this.session = null;
    }
  }
}

async function main() {
  const v2 = JSON.parse(fs.readFileSync('library_v2.json', 'utf8'));

  const missingJobs = { ar: [], fr: [], es: [] };

  for (const item of v2) {
    // Arabic
    const arMp3 = path.resolve(item.audio_ar);
    if (!fs.existsSync(arMp3) || fs.statSync(arMp3).size < 1000) {
      const { line, emoDesc } = cleanScriptLine(item.phrase_ar);
      missingJobs.ar.push({
        id: item.id,
        baseOutPath: arMp3.replace(/\.mp3$/, ''),
        prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this Arabic script line with that exact emotional football commentator passion (do NOT speak the direction): "${line}"`,
      });
    }

    // French
    const frMp3 = path.resolve(item.audio_fr);
    if (!fs.existsSync(frMp3) || fs.statSync(frMp3).size < 1000) {
      const { line, emoDesc } = cleanScriptLine(item.phrase_fr);
      missingJobs.fr.push({
        id: item.id,
        baseOutPath: frMp3.replace(/\.mp3$/, ''),
        prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this French script line with authentic French live football commentator passion (do NOT speak the direction): "${line}"`,
      });
    }

    // Spanish
    const esMp3 = path.resolve(item.audio_es);
    if (!fs.existsSync(esMp3) || fs.statSync(esMp3).size < 1000) {
      const { line, emoDesc } = cleanScriptLine(item.phrase_es);
      missingJobs.es.push({
        id: item.id,
        baseOutPath: esMp3.replace(/\.mp3$/, ''),
        prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this Spanish script line with authentic Latin American/Spanish live football commentator passion (do NOT speak the direction): "${line}"`,
      });
    }
  }

  const totalMissing = missingJobs.ar.length + missingJobs.fr.length + missingJobs.es.length;
  console.log(`Audited missing audio clips: AR=${missingJobs.ar.length}, FR=${missingJobs.fr.length}, ES=${missingJobs.es.length} (Total: ${totalMissing})`);

  if (totalMissing === 0) {
    console.log('All audio clips are already 100% complete!');
    return;
  }

  let completed = 0;
  const workerCountMap = { ar: 2, fr: 3, es: 2 };
  const workerPromises = [];

  for (const lang of ['ar', 'fr', 'es']) {
    const queue = missingJobs[lang];
    const cfg = langConfigs[lang];
    const numWorkers = workerCountMap[lang];

    for (let w = 0; w < numWorkers; w++) {
      workerPromises.push((async () => {
        const worker = new LiveVoiceWorker(cfg.voice, cfg.instruction);
        try {
          while (queue.length > 0) {
            const job = queue.shift();
            if (!job) break;
            const pcm = await worker.synthesize(job.prompt);
            await savePcmAsWavAndMp3(pcm, job.baseOutPath);
            completed++;
            console.log(`[${lang.toUpperCase()}] ${job.id}.mp3 generated (${completed}/${totalMissing})`);
          }
        } finally {
          worker.close();
        }
      })());
    }
  }

  await Promise.all(workerPromises);
  console.log('All missing audio clips successfully generated!');
}

main().catch(err => {
  console.error('Fatal error in generate-missing:', err);
  process.exit(1);
});
