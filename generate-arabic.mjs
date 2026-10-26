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

const GITHUB_BASE = 'https://imad-os.github.io/f_assets/';

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

class ArabicVoiceWorker {
  constructor(voiceName, genderDesc) {
    this.voiceName = voiceName;
    this.genderDesc = genderDesc;
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
          parts: [{
            text: `You are a world-class ${this.genderDesc} Arabic football commentator in a recording studio. Read aloud ONLY the script line in fluent native Arabic with dramatic, passionate stadium commentary emotion (Issam Chaouali / Hafid Derradji style). Sound 100% human and authentic. NEVER read acting directions, brackets, or stage notes out loud. Speak ONLY the Arabic line without extra words.`,
          }],
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

  const arBaseDir = path.resolve('audio/ar');
  const arMaleDir = path.resolve('audio/ar/male');
  const arFemaleDir = path.resolve('audio/ar/female');

  [arBaseDir, arMaleDir, arFemaleDir].forEach(d => fs.mkdirSync(d, { recursive: true }));

  const maleJobs = [];
  const femaleJobs = [];

  for (const item of v2) {
    const { line, emoDesc } = cleanScriptLine(item.phrase_ar);
    const prompt = `Acting direction: (${emoDesc}). Read aloud ONLY this Arabic script line with that exact passionate live football commentator emotion (do NOT speak the direction): "${line}"`;

    // Male Arabic: check both audio/ar/male/ and audio/ar/
    const maleMp3 = path.join(arMaleDir, `${item.id}.mp3`);
    const baseMp3 = path.join(arBaseDir, `${item.id}.mp3`);

    const hasMale = (fs.existsSync(maleMp3) && fs.statSync(maleMp3).size > 2000) ||
                    (fs.existsSync(baseMp3) && fs.statSync(baseMp3).size > 2000);

    if (!hasMale) {
      maleJobs.push({
        id: item.id,
        prompt,
        maleBase: path.join(arMaleDir, item.id),
        baseBase: path.join(arBaseDir, item.id),
      });
    } else {
      // Ensure sync between male/ and root ar/
      if (fs.existsSync(baseMp3) && !fs.existsSync(maleMp3)) {
        fs.copyFileSync(baseMp3, maleMp3);
        const baseWav = baseMp3.replace(/\.mp3$/, '.wav');
        const maleWav = maleMp3.replace(/\.mp3$/, '.wav');
        if (fs.existsSync(baseWav)) fs.copyFileSync(baseWav, maleWav);
      } else if (fs.existsSync(maleMp3) && !fs.existsSync(baseMp3)) {
        fs.copyFileSync(maleMp3, baseMp3);
        const maleWav = maleMp3.replace(/\.mp3$/, '.wav');
        const baseWav = baseMp3.replace(/\.mp3$/, '.wav');
        if (fs.existsSync(maleWav)) fs.copyFileSync(maleWav, baseWav);
      }
    }

    // Female Arabic: check audio/ar/female/
    const femaleMp3 = path.join(arFemaleDir, `${item.id}.mp3`);
    if (!fs.existsSync(femaleMp3) || fs.statSync(femaleMp3).size < 2000) {
      femaleJobs.push({
        id: item.id,
        prompt,
        femaleBase: path.join(arFemaleDir, item.id),
      });
    }
  }

  console.log(`Arabic Audio Audit: Male missing=${maleJobs.length}/75, Female missing=${femaleJobs.length}/75`);

  let completed = 0;
  const total = maleJobs.length + femaleJobs.length;

  if (total === 0) {
    console.log('All Arabic male and female audio files are already 100% complete!');
  } else {
    // 2 male workers (Fenrir) and 2 female workers (Kore)
    const workers = [
      // 2 Male workers
      (async () => {
        const worker = new ArabicVoiceWorker('Fenrir', 'male');
        try {
          while (maleJobs.length > 0) {
            const job = maleJobs.shift();
            if (!job) break;
            const pcm = await worker.synthesize(job.prompt);
            await savePcmAsWavAndMp3(pcm, job.maleBase);
            // Copy to base ar/ as well
            fs.copyFileSync(`${job.maleBase}.mp3`, `${job.baseBase}.mp3`);
            fs.copyFileSync(`${job.maleBase}.wav`, `${job.baseBase}.wav`);
            completed++;
            console.log(`[AR MALE] ${job.id}.mp3 generated (${completed}/${total})`);
          }
        } finally {
          worker.close();
        }
      })(),
      (async () => {
        const worker = new ArabicVoiceWorker('Fenrir', 'male');
        try {
          while (maleJobs.length > 0) {
            const job = maleJobs.shift();
            if (!job) break;
            const pcm = await worker.synthesize(job.prompt);
            await savePcmAsWavAndMp3(pcm, job.maleBase);
            fs.copyFileSync(`${job.maleBase}.mp3`, `${job.baseBase}.mp3`);
            fs.copyFileSync(`${job.maleBase}.wav`, `${job.baseBase}.wav`);
            completed++;
            console.log(`[AR MALE] ${job.id}.mp3 generated (${completed}/${total})`);
          }
        } finally {
          worker.close();
        }
      })(),
      // 2 Female workers
      (async () => {
        const worker = new ArabicVoiceWorker('Kore', 'female');
        try {
          while (femaleJobs.length > 0) {
            const job = femaleJobs.shift();
            if (!job) break;
            const pcm = await worker.synthesize(job.prompt);
            await savePcmAsWavAndMp3(pcm, job.femaleBase);
            completed++;
            console.log(`[AR FEMALE] ${job.id}.mp3 generated (${completed}/${total})`);
          }
        } finally {
          worker.close();
        }
      })(),
      (async () => {
        const worker = new ArabicVoiceWorker('Kore', 'female');
        try {
          while (femaleJobs.length > 0) {
            const job = femaleJobs.shift();
            if (!job) break;
            const pcm = await worker.synthesize(job.prompt);
            await savePcmAsWavAndMp3(pcm, job.femaleBase);
            completed++;
            console.log(`[AR FEMALE] ${job.id}.mp3 generated (${completed}/${total})`);
          }
        } finally {
          worker.close();
        }
      })(),
    ];

    await Promise.all(workers);
    console.log('Successfully finished generating all Arabic audio files!');
  }

  // Update library_v2.json with both male and female Arabic paths and full URLs
  const updated = v2.map(item => {
    const maleRel = `audio/ar/male/${item.id}.mp3`;
    const femaleRel = `audio/ar/female/${item.id}.mp3`;
    const maleFull = `${GITHUB_BASE}${maleRel}`;
    const femaleFull = `${GITHUB_BASE}${femaleRel}`;

    return {
      ...item,
      audio_ar: maleRel,
      audio_ar_male: maleRel,
      audio_ar_female: femaleRel,
      audio_full_url_ar: maleFull,
      audio_full_url_ar_male: maleFull,
      audio_full_url_ar_female: femaleFull,
      translations: {
        ...item.translations,
        ar: {
          ...item.translations.ar,
          audio_url: maleRel,
          audio_url_male: maleRel,
          audio_url_female: femaleRel,
          wav_url: `audio/ar/male/${item.id}.wav`,
          wav_url_male: `audio/ar/male/${item.id}.wav`,
          wav_url_female: `audio/ar/female/${item.id}.wav`,
          audio_full_url: maleFull,
          audio_full_url_male: maleFull,
          audio_full_url_female: femaleFull,
        },
      },
    };
  });

  fs.writeFileSync('library_v2.json', JSON.stringify(updated, null, 2) + '\n');

  // Update audio/ar/ manifests
  const arManifest = updated.map(i => ({
    id: i.id,
    event: i.event,
    phrase: i.phrase_ar,
    audio_url: i.audio_ar_male,
    audio_url_male: i.audio_ar_male,
    audio_url_female: i.audio_ar_female,
    audio_full_url_male: i.audio_full_url_ar_male,
    audio_full_url_female: i.audio_full_url_ar_female,
  }));
  fs.writeFileSync('audio/ar/library.json', JSON.stringify(arManifest, null, 2) + '\n');
  fs.writeFileSync('audio/ar/male/library.json', JSON.stringify(arManifest.map(i => ({ id: i.id, event: i.event, phrase: i.phrase, audio_url: i.audio_url_male, audio_full_url: i.audio_full_url_male })), null, 2) + '\n');
  fs.writeFileSync('audio/ar/female/library.json', JSON.stringify(arManifest.map(i => ({ id: i.id, event: i.event, phrase: i.phrase, audio_url: i.audio_url_female, audio_full_url: i.audio_full_url_female })), null, 2) + '\n');

  console.log('library_v2.json and Arabic manifests successfully updated with male and female Arabic tracks!');
}

main().catch(err => {
  console.error('Fatal in generate-arabic:', err);
  process.exit(1);
});
