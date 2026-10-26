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

// Fix phonetics per language:
// French: replace "buuuut" with "But !" (tag instructs extended scream)
// Arabic: replace "جول / جوووول" with "هدف / هـــــدف" (avoids "Joal" pronunciation)
function cleanPhraseForTts(phrase, lang) {
  const regex = /\[([^\]]+)\]\s*([^\[]*)/g;
  const spoken = [];
  const emotions = [];
  let match;

  while ((match = regex.exec(phrase)) !== null) {
    const emo = match[1].trim();
    let txt = match[2].trim();
    if (!txt) continue;
    emotions.push(emo);

    if (lang === 'fr') {
      // Fix 'buuuut' -> 'But !'
      txt = txt.replace(/\bbu+t+\b/gi, 'But !');
      txt = txt.replace(/\bBu+t+\b/g, 'But !');
    } else if (lang === 'ar') {
      // Fix 'جول / جوووول' -> 'هـــــدف'
      txt = txt.replace(/ج+و+ل+/g, 'هـــــدف');
    }

    spoken.push(txt);
  }

  let line = spoken.length > 0 ? spoken.join(' ') : phrase.replace(/\[[^\]]+\]/g, '').trim();
  if (lang === 'fr') {
    line = line.replace(/\bbu+t+\b/gi, 'But !');
  } else if (lang === 'ar') {
    line = line.replace(/ج+و+ل+/g, 'هـــــدف');
  }

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

  // Ensure directories exist
  const dirs = [
    'audio/fr',
    'audio/es',
    'audio/ar',
    'audio/ar/male',
    'audio/ar/female',
  ];
  dirs.forEach(d => fs.mkdirSync(path.resolve(d), { recursive: true }));

  const forceRegenIds = new Set();

  const jobs = {
    fr: [],
    es: [],
    ar_male: [],
    ar_female: [],
  };

  for (const item of v2) {
    // 1. French
    const frBase = path.resolve('audio/fr', item.id);
    const frMp3 = `${frBase}.mp3`;
    const needFr = forceRegenIds.has(item.id) || !fs.existsSync(frMp3) || fs.statSync(frMp3).size < 2000;
    if (needFr) {
      const { line, emoDesc } = cleanPhraseForTts(item.phrase_fr, 'fr');
      jobs.fr.push({
        id: item.id,
        baseOutPath: frBase,
        prompt: `Acting direction: (${emoDesc}; shout "But !" with an extended triumphant French stadium goal scream if calling a goal). Read aloud ONLY this French script line: "${line}"`,
      });
    }

    // 2. Spanish
    const esBase = path.resolve('audio/es', item.id);
    const esMp3 = `${esBase}.mp3`;
    const needEs = !fs.existsSync(esMp3) || fs.statSync(esMp3).size < 2000;
    if (needEs) {
      const { line, emoDesc } = cleanPhraseForTts(item.phrase_es, 'es');
      jobs.es.push({
        id: item.id,
        baseOutPath: esBase,
        prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this Spanish script line with explosive Latin American/Spanish stadium commentator passion: "${line}"`,
      });
    }

    // 3. Arabic Male
    const arBase = path.resolve('audio/ar', item.id);
    const arMaleBase = path.resolve('audio/ar/male', item.id);
    const arMp3 = `${arBase}.mp3`;
    const needArMale = forceRegenIds.has(item.id) || !fs.existsSync(arMp3) || fs.statSync(arMp3).size < 2000;
    if (needArMale) {
      const { line, emoDesc } = cleanPhraseForTts(item.phrase_ar, 'ar');
      jobs.ar_male.push({
        id: item.id,
        baseOutPath: arMaleBase,
        copyToBase: arBase,
        prompt: `Acting direction: (${emoDesc}; pronounce goal shouts as "هـــــدف!" with triumphant Arabic commentator passion "Hadaaaf!"). Read aloud ONLY this Arabic script line: "${line}"`,
      });
    } else {
      // Sync to audio/ar/male/ if already present in audio/ar/
      const maleMp3 = `${arMaleBase}.mp3`;
      if (!fs.existsSync(maleMp3) && fs.existsSync(arMp3)) {
        fs.copyFileSync(arMp3, maleMp3);
        const wav = `${arBase}.wav`;
        if (fs.existsSync(wav)) fs.copyFileSync(wav, `${arMaleBase}.wav`);
      }
    }

    // 4. Arabic Female
    const arFemaleBase = path.resolve('audio/ar/female', item.id);
    const arFemaleMp3 = `${arFemaleBase}.mp3`;
    const needArFemale = forceRegenIds.has(item.id) || !fs.existsSync(arFemaleMp3) || fs.statSync(arFemaleMp3).size < 2000;
    if (needArFemale) {
      const { line, emoDesc } = cleanPhraseForTts(item.phrase_ar, 'ar');
      jobs.ar_female.push({
        id: item.id,
        baseOutPath: arFemaleBase,
        prompt: `Acting direction: (${emoDesc}; pronounce goal shouts as "هـــــدف!" with vibrant female Arabic stadium commentator passion "Hadaaaf!"). Read aloud ONLY this Arabic script line: "${line}"`,
      });
    }
  }

  const totalJobs = jobs.fr.length + jobs.es.length + jobs.ar_male.length + jobs.ar_female.length;
  console.log(`Audited Audio Jobs to Run: FR=${jobs.fr.length}, ES=${jobs.es.length}, AR_MALE=${jobs.ar_male.length}, AR_FEMALE=${jobs.ar_female.length} (Total: ${totalJobs})`);

  let completed = 0;

  const configs = {
    fr: {
      voice: 'Kore',
      instruction: 'You are a dynamic native French football commentator. Pronounce goal calls with "But !" with an extended triumphant goal shout ("Buuuuut !"). NEVER say stage directions.',
      workers: 3,
    },
    es: {
      voice: 'Puck',
      instruction: 'You are an explosive native Spanish football commentator. Speak with authentic Latin American/Spanish stadium commentator passion. NEVER say stage directions.',
      workers: 2,
    },
    ar_male: {
      voice: 'Fenrir',
      instruction: 'You are a passionate male Arabic football commentator (Issam Chaouali / Hafid Derradji style). Pronounce goal shouts as "هـــــدف!" (Hadaaaaf!). Speak ONLY the Arabic line.',
      workers: 2,
    },
    ar_female: {
      voice: 'Kore',
      instruction: 'You are an enthusiastic female Arabic football commentator. Pronounce goal shouts as "هـــــدف!" (Hadaaaaf!). Speak ONLY the Arabic line with vibrant commentator passion.',
      workers: 2,
    },
  };

  const workerPromises = [];

  for (const [key, q] of Object.entries(jobs)) {
    const cfg = configs[key];
    for (let w = 0; w < cfg.workers; w++) {
      workerPromises.push((async () => {
        const worker = new LiveVoiceWorker(cfg.voice, cfg.instruction);
        try {
          while (q.length > 0) {
            const job = q.shift();
            if (!job) break;
            const pcm = await worker.synthesize(job.prompt);
            await savePcmAsWavAndMp3(pcm, job.baseOutPath);
            if (job.copyToBase) {
              fs.copyFileSync(`${job.baseOutPath}.mp3`, `${job.copyToBase}.mp3`);
              fs.copyFileSync(`${job.baseOutPath}.wav`, `${job.copyToBase}.wav`);
            }
            completed++;
            console.log(`[${key.toUpperCase()}] ${job.id}.mp3 generated (${completed}/${totalJobs})`);
          }
        } finally {
          worker.close();
        }
      })());
    }
  }

  await Promise.all(workerPromises);

  // Update library_v2.json (NOTE: user said "Dont change library json", library.json is untouched!)
  const updatedV2 = v2.map(item => {
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

  fs.writeFileSync('library_v2.json', JSON.stringify(updatedV2, null, 2) + '\n');
  console.log('Successfully completed all audio generation with phonetic pronunciation fixes!');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
