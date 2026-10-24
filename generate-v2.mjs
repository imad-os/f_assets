import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { GoogleGenAI, Modality, Type } from '@google/genai';

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

// Translation prompt function with retry and fallback
async function translateBatch(items, attempt = 1) {
  const model = attempt <= 2 ? 'gemini-3.8-flash' : 'gemini-3.1-flash-lite';
  const prompt = `You are a professional multilingual football (soccer) commentator and translator.
Translate the following football commentator phrases into Arabic (ar), French (fr), and Spanish (es).
CRITICAL RULES:
1. Preserve every bracketed emotional direction tag exactly (e.g. [excited], [shouting], [screaming], [laughs], [breathless], [gasp], [serious], [annoyed], [calm], [surprised], [tense], [whispering], [dramatic], [focused], [intense], [amazed], [impressed], [urgent], [disappointed], [groans], [cheerful], [energetic], [enthusiastic], [matter-of-fact], [relaxed], [thoughtful], [emotional]).
2. Translate the commentary text so it sounds completely authentic and natural to real live football broadcasts in each language:
   - Arabic: Authentic, passionate Arabic football commentary (like Issam Chaouali / Hafid Derradji style).
   - French: Authentic, vivid French football commentary (like Canal+ / Grégoire Margotton style).
   - Spanish: Authentic, passionate Latin American / Spanish football commentary (like ¡Goooool!, ¡Qué definición!).

Phrases to translate:
${JSON.stringify(items.map(it => ({ id: it.id, phrase: it.phrase })), null, 2)}
`;

  try {
    const res = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.STRING },
              ar: { type: Type.STRING, description: 'Arabic translation with bracketed emotion tags' },
              fr: { type: Type.STRING, description: 'French translation with bracketed emotion tags' },
              es: { type: Type.STRING, description: 'Spanish translation with bracketed emotion tags' },
            },
            required: ['id', 'ar', 'fr', 'es'],
          },
        },
      },
    });

    return JSON.parse(res.text);
  } catch (err) {
    if (attempt < 5) {
      const waitMs = 2000 * attempt;
      console.log(`Translation retry #${attempt} in ${waitMs}ms...`);
      await new Promise(r => setTimeout(r, waitMs));
      return translateBatch(items, attempt + 1);
    }
    throw err;
  }
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
      '-f',
      's16le',
      '-ar',
      '24000',
      '-ac',
      '1',
      '-i',
      rawPath,
      wavPath,
    ]);
    await execFileAsync('ffmpeg', [
      '-y',
      '-f',
      's16le',
      '-ar',
      '24000',
      '-ac',
      '1',
      '-i',
      rawPath,
      '-codec:a',
      'libmp3lame',
      '-qscale:a',
      '2',
      mp3Path,
    ]);
  } finally {
    if (fs.existsSync(rawPath)) fs.unlinkSync(rawPath);
  }
}

class MultilingualVoiceWorker {
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
          parts: [
            {
              text: `You are a world-class football commentator recording voice tracks in a studio. ${this.langInstruction} For each prompt, perform aloud ONLY the quoted script line using the exact vocal emotion, shouting, screaming, laughing, whispering, or breathless energy described in the acting direction. Sound 100% native, passionate, and authentic. NEVER read the acting direction brackets or words out loud. Speak ONLY the line in that native language without extra preamble.`,
            },
          ],
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
      if (!this.session || this.turnsUsed >= 8) {
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
        await new Promise((r) => setTimeout(r, 1000 * attempt));
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

async function main() {
  const rawLib = JSON.parse(fs.readFileSync('library.json', 'utf8'));

  // Ensure directories exist
  const audioDir = path.resolve('audio');
  const arDir = path.resolve('audio/ar');
  const frDir = path.resolve('audio/fr');
  const esDir = path.resolve('audio/es');
  const enMaleDir = path.resolve('audio/en/male');
  const enFemaleDir = path.resolve('audio/en/female');

  [audioDir, arDir, frDir, esDir, enMaleDir, enFemaleDir].forEach(d => fs.mkdirSync(d, { recursive: true }));

  // Copy or link existing en male/female into audio/en
  if (fs.existsSync('en/male')) {
    const existingMale = fs.readdirSync('en/male');
    for (const f of existingMale) {
      const src = path.join('en/male', f);
      const dst = path.join(enMaleDir, f);
      if (!fs.existsSync(dst)) fs.copyFileSync(src, dst);
    }
  }
  if (fs.existsSync('en/female')) {
    const existingFemale = fs.readdirSync('en/female');
    for (const f of existingFemale) {
      const src = path.join('en/female', f);
      const dst = path.join(enFemaleDir, f);
      if (!fs.existsSync(dst)) fs.copyFileSync(src, dst);
    }
  }

  // Assign IDs to items
  const eventCounters = {};
  const itemsWithIds = rawLib.map(item => {
    const ev = item.event.toUpperCase();
    eventCounters[ev] = (eventCounters[ev] || 0) + 1;
    return {
      id: `${ev}_${eventCounters[ev]}`,
      event: ev,
      phrase_en: item.phrase,
      audio_en_male: `audio/en/male/${ev}_${eventCounters[ev]}.mp3`,
      audio_en_female: `audio/en/female/${ev}_${eventCounters[ev]}.mp3`,
    };
  });

  // Check if translation cache exists
  const cacheFile = 'translations_cache.json';
  let translations = {};
  if (fs.existsSync(cacheFile)) {
    try {
      translations = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
    } catch {}
  }

  const missingTranslations = itemsWithIds.filter(it => !translations[it.id]);
  if (missingTranslations.length > 0) {
    console.log(`Translating ${missingTranslations.length} phrases in batches...`);
    const BATCH_SIZE = 15;
    for (let i = 0; i < missingTranslations.length; i += BATCH_SIZE) {
      const slice = missingTranslations.slice(i, i + BATCH_SIZE);
      const translatedBatch = await translateBatch(slice.map(s => ({ id: s.id, phrase: s.phrase_en })));
      for (const t of translatedBatch) {
        translations[t.id] = { ar: t.ar, fr: t.fr, es: t.es };
      }
      fs.writeFileSync(cacheFile, JSON.stringify(translations, null, 2));
      console.log(`Translated ${Math.min(i + BATCH_SIZE, missingTranslations.length)}/${missingTranslations.length}`);
    }
  } else {
    console.log('All translations already cached.');
  }

  // Build jobs for TTS
  const jobs = [];
  for (const item of itemsWithIds) {
    const t = translations[item.id];
    if (!t) continue;

    // Arabic
    const arBase = path.join(arDir, item.id);
    if (!fs.existsSync(`${arBase}.mp3`) || fs.statSync(`${arBase}.mp3`).size < 1000) {
      const { line, emoDesc } = cleanScriptLine(t.ar);
      jobs.push({
        lang: 'ar',
        id: item.id,
        baseOutPath: arBase,
        prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this Arabic script line with that exact emotional football commentator passion (do NOT speak the direction): "${line}"`,
      });
    }

    // French
    const frBase = path.join(frDir, item.id);
    if (!fs.existsSync(`${frBase}.mp3`) || fs.statSync(`${frBase}.mp3`).size < 1000) {
      const { line, emoDesc } = cleanScriptLine(t.fr);
      jobs.push({
        lang: 'fr',
        id: item.id,
        baseOutPath: frBase,
        prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this French script line with authentic French live football commentator passion (do NOT speak the direction): "${line}"`,
      });
    }

    // Spanish
    const esBase = path.join(esDir, item.id);
    if (!fs.existsSync(`${esBase}.mp3`) || fs.statSync(`${esBase}.mp3`).size < 1000) {
      const { line, emoDesc } = cleanScriptLine(t.es);
      jobs.push({
        lang: 'es',
        id: item.id,
        baseOutPath: esBase,
        prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this Spanish script line with authentic Latin American/Spanish live football commentator passion (do NOT speak the direction): "${line}"`,
      });
    }
  }

  console.log(`TTS generation jobs needed: ${jobs.length}`);

  const langConfigs = {
    ar: { voice: 'Fenrir', instruction: 'You speak fluent native Arabic with dramatic, passionate Arabic football commentary style (Issam Chaouali style).' },
    fr: { voice: 'Kore', instruction: 'You speak fluent native French with vivid, dynamic French sports broadcasting commentary style.' },
    es: { voice: 'Puck', instruction: 'You speak fluent native Spanish with high-energy, explosive Spanish football commentary style (¡Gooool!).' },
  };

  const jobsByLang = { ar: [], fr: [], es: [] };
  for (const j of jobs) {
    jobsByLang[j.lang].push(j);
  }

  let completedJobs = 0;
  const totalJobs = jobs.length;

  const workerPromises = [];
  for (const [lang, langJobs] of Object.entries(jobsByLang)) {
    const numWorkers = 2; // 2 workers per language = 6 concurrent workers
    for (let w = 0; w < numWorkers; w++) {
      workerPromises.push((async () => {
        const worker = new MultilingualVoiceWorker(langConfigs[lang].voice, langConfigs[lang].instruction);
        try {
          while (langJobs.length > 0) {
            const job = langJobs.shift();
            if (!job) break;
            const pcm = await worker.synthesize(job.prompt);
            await savePcmAsWavAndMp3(pcm, job.baseOutPath);
            completedJobs++;
            console.log(`[${job.lang.toUpperCase()}] ${job.id}.mp3 done (${completedJobs}/${totalJobs})`);
          }
        } finally {
          worker.close();
        }
      })());
    }
  }

  await Promise.all(workerPromises);

  // Now assemble library_v2.json
  const v2Library = itemsWithIds.map(item => {
    const t = translations[item.id] || {};
    return {
      id: item.id,
      event: item.event,
      translations: {
        en: {
          phrase: item.phrase_en,
          audio_url_male: item.audio_en_male,
          audio_url_female: item.audio_en_female,
          audio_url: item.audio_en_male,
        },
        ar: {
          phrase: t.ar || '',
          audio_url: `audio/ar/${item.id}.mp3`,
          wav_url: `audio/ar/${item.id}.wav`,
        },
        fr: {
          phrase: t.fr || '',
          audio_url: `audio/fr/${item.id}.mp3`,
          wav_url: `audio/fr/${item.id}.wav`,
        },
        es: {
          phrase: t.es || '',
          audio_url: `audio/es/${item.id}.mp3`,
          wav_url: `audio/es/${item.id}.wav`,
        },
      },
      phrase_en: item.phrase_en,
      phrase_ar: t.ar || '',
      phrase_fr: t.fr || '',
      phrase_es: t.es || '',
      audio_en: item.audio_en_male,
      audio_en_female: item.audio_en_female,
      audio_ar: `audio/ar/${item.id}.mp3`,
      audio_fr: `audio/fr/${item.id}.mp3`,
      audio_es: `audio/es/${item.id}.mp3`,
    };
  });

  fs.writeFileSync('library_v2.json', JSON.stringify(v2Library, null, 2) + '\n');
  fs.writeFileSync('library-v2.json', JSON.stringify(v2Library, null, 2) + '\n');

  console.log('Successfully completed translations and audio generation for library_v2.json!');
}

main().catch(err => {
  console.error('Fatal in generate-v2:', err);
  process.exit(1);
});
