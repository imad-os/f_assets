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

function updateLibraryTextAndSync() {
  const v2 = JSON.parse(fs.readFileSync('library_v2.json', 'utf8'));
  const arChangedIds = [];
  const frChangedIds = [];

  for (const item of v2) {
    // Check and fix Arabic "جوووووول" -> "هدف"
    if (/ج+و+ل+/u.test(item.phrase_ar)) {
      item.phrase_ar = item.phrase_ar.replace(/ج+و+ل+/gu, 'هدف');
      if (item.translations?.ar) {
        item.translations.ar.phrase = item.phrase_ar;
      }
      arChangedIds.push(item.id);
    }

    // Check and fix French "Buuuuut" -> "But"
    if (/\bbu{2,}t+\b/i.test(item.phrase_fr)) {
      item.phrase_fr = item.phrase_fr.replace(/\bBu{2,}t+\b/g, 'But').replace(/\bbu{2,}t+\b/g, 'but');
      if (item.translations?.fr) {
        item.translations.fr.phrase = item.phrase_fr;
      }
      frChangedIds.push(item.id);
    }
  }

  // Ensure directories exist
  ['audio/ar', 'audio/ar/male', 'audio/ar/female', 'audio/fr', 'audio/es'].forEach((d) =>
    fs.mkdirSync(path.resolve(d), { recursive: true })
  );

  // Sync existing audio/ar/*.mp3 <-> audio/ar/male/*.mp3
  for (const item of v2) {
    const rootMp3 = path.resolve('audio/ar', `${item.id}.mp3`);
    const rootWav = path.resolve('audio/ar', `${item.id}.wav`);
    const maleMp3 = path.resolve('audio/ar/male', `${item.id}.mp3`);
    const maleWav = path.resolve('audio/ar/male', `${item.id}.wav`);

    if (fs.existsSync(rootMp3) && !fs.existsSync(maleMp3)) {
      fs.copyFileSync(rootMp3, maleMp3);
    } else if (fs.existsSync(maleMp3) && !fs.existsSync(rootMp3)) {
      fs.copyFileSync(maleMp3, rootMp3);
    }
    if (fs.existsSync(rootWav) && !fs.existsSync(maleWav)) {
      fs.copyFileSync(rootWav, maleWav);
    } else if (fs.existsSync(maleWav) && !fs.existsSync(rootWav)) {
      fs.copyFileSync(maleWav, rootWav);
    }
  }

  fs.writeFileSync('library_v2.json', JSON.stringify(v2, null, 2) + '\n');
  return { v2, arChangedIds, frChangedIds };
}

function parseSpokenAndEmotion(phrase) {
  const regex = /\[([^\]]+)\]\s*([^\[]*)/g;
  const spoken = [];
  const emotions = [];
  let match;

  while ((match = regex.exec(phrase)) !== null) {
    const emo = match[1].trim();
    const txt = match[2].trim();
    if (!txt) continue;
    emotions.push(emo);
    spoken.push(txt);
  }

  const line =
    spoken.length > 0
      ? spoken.join(' ')
      : phrase.replace(/\[[^\]]+\]/g, '').trim();
  const emoDesc = emotions.length > 0 ? emotions.join(', ') : 'excited, passionate';
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

class LiveVoiceWorker {
  constructor(voiceName, systemText) {
    this.voiceName = voiceName;
    this.systemText = systemText;
    this.session = null;
    this.turnsUsed = 0;
    this.chunks = [];
    this.resolveTurn = null;
    this.rejectTurn = null;
  }

  async connect() {
    if (this.session) {
      try {
        this.session.close();
      } catch {}
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
          parts: [{ text: this.systemText }],
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
            fn(new Error('Live session closed'));
          }
          this.session = null;
        },
      },
    });
  }

  async synthesize(prompt, attempt = 1) {
    try {
      if (!this.session || this.turnsUsed >= 5) {
        await this.connect();
      }
      this.chunks = [];
      this.turnsUsed++;

      const pcmBuffer = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          this.resolveTurn = null;
          this.rejectTurn = null;
          reject(new Error('Turn timeout after 20s'));
        }, 20000);

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
        await new Promise((r) => setTimeout(r, 600 * attempt));
        return this.synthesize(prompt, attempt + 1);
      }
      throw err;
    }
  }

  close() {
    if (this.session) {
      try {
        this.session.close();
      } catch {}
      this.session = null;
    }
  }
}

async function run() {
  const mode = process.argv[2] || 'audit';
  const maxBatch = Number(process.argv[3]) || 30;

  const { v2, arChangedIds, frChangedIds } = updateLibraryTextAndSync();
  if (arChangedIds.length > 0 || frChangedIds.length > 0) {
    console.log('Updated library_v2.json text:', { arChangedIds, frChangedIds });
  }

  const arGoalIds = new Set(['GOAL_1', 'GOAL_2', 'GOAL_4']);

  if (mode === 'ar-es') {
    const arMaleJobs = [];
    const arFemaleJobs = [];
    const esJobs = [];

    for (const item of v2) {
      // Regenerate GOAL_1, GOAL_2, GOAL_4 for Arabic (which had جووووول) or any missing Arabic
      const arMaleBase = path.resolve('audio/ar/male', item.id);
      const arRootBase = path.resolve('audio/ar', item.id);
      const arFemaleBase = path.resolve('audio/ar/female', item.id);

      const forceAr = arGoalIds.has(item.id) && !fs.existsSync(`${arMaleBase}.fixed_hadaf`);
      if (forceAr || !fs.existsSync(`${arMaleBase}.mp3`) || fs.statSync(`${arMaleBase}.mp3`).size < 2000) {
        const { line, emoDesc } = parseSpokenAndEmotion(item.phrase_ar);
        const extraGoalNote = item.event === 'GOAL'
          ? 'VERY IMPORTANT: Shout the word "هدف" with extreme excitement, stretching and elongating the vowel ("هـــــــــدف!") just like an English commentator shouting "Gooooooal!"'
          : '';
        arMaleJobs.push({
          id: item.id,
          baseOutPath: arMaleBase,
          copyToPath: arRootBase,
          markerFile: forceAr ? `${arMaleBase}.fixed_hadaf` : null,
          prompt: `Acting direction: (${emoDesc}). ${extraGoalNote}\nRead aloud ONLY this Arabic script line with maximum stadium passion (do NOT speak the acting direction): "${line}"`,
        });
      }

      const forceArFem = arGoalIds.has(item.id) && !fs.existsSync(`${arFemaleBase}.fixed_hadaf`);
      if (forceArFem || !fs.existsSync(`${arFemaleBase}.mp3`) || fs.statSync(`${arFemaleBase}.mp3`).size < 2000) {
        const { line, emoDesc } = parseSpokenAndEmotion(item.phrase_ar);
        const extraGoalNote = item.event === 'GOAL'
          ? 'VERY IMPORTANT: Shout the word "هدف" with extreme excitement, stretching and elongating the vowel ("هـــــــــدف!") just like an English commentator shouting "Gooooooal!"'
          : '';
        arFemaleJobs.push({
          id: item.id,
          baseOutPath: arFemaleBase,
          markerFile: forceArFem ? `${arFemaleBase}.fixed_hadaf` : null,
          prompt: `Acting direction: (${emoDesc}). ${extraGoalNote}\nRead aloud ONLY this Arabic script line with maximum stadium passion (do NOT speak the acting direction): "${line}"`,
        });
      }

      // Check Spanish missing
      const esBase = path.resolve('audio/es', item.id);
      if (!fs.existsSync(`${esBase}.mp3`) || fs.statSync(`${esBase}.mp3`).size < 2000) {
        const { line, emoDesc } = parseSpokenAndEmotion(item.phrase_es);
        esJobs.push({
          id: item.id,
          baseOutPath: esBase,
          prompt: `Acting direction: (${emoDesc}). Read aloud ONLY this Spanish script line with explosive Spanish football commentator passion (do NOT speak the acting direction): "${line}"`,
        });
      }
    }

    console.log(`AR-ES Queue: AR_MALE=${arMaleJobs.length}, AR_FEMALE=${arFemaleJobs.length}, ES=${esJobs.length}`);

    const workers = [];
    async function runQueue(queue, voice, sysText, numWorkers, tag) {
      for (let w = 0; w < numWorkers; w++) {
        workers.push(
          (async () => {
            const worker = new LiveVoiceWorker(voice, sysText);
            try {
              while (queue.length > 0) {
                const job = queue.shift();
                if (!job) break;
                const pcm = await worker.synthesize(job.prompt);
                await savePcmAsWavAndMp3(pcm, job.baseOutPath);
                if (job.copyToPath) {
                  fs.copyFileSync(`${job.baseOutPath}.mp3`, `${job.copyToPath}.mp3`);
                  fs.copyFileSync(`${job.baseOutPath}.wav`, `${job.copyToPath}.wav`);
                }
                if (job.markerFile) {
                  fs.writeFileSync(job.markerFile, 'ok');
                }
                console.log(`[${tag}] ${job.id}.mp3 done`);
              }
            } finally {
              worker.close();
            }
          })()
        );
      }
    }

    if (arMaleJobs.length > 0) {
      runQueue(
        arMaleJobs,
        'Fenrir',
        'You are a legendary male Arabic football commentator (Issam Chaouali style). When shouting "هدف", stretch and elongate the word with extreme stadium excitement ("هــــــــدف!") like "Gooooooal!". Speak ONLY the Arabic line.',
        2,
        'AR_MALE'
      );
    }
    if (arFemaleJobs.length > 0) {
      runQueue(
        arFemaleJobs,
        'Kore',
        'You are a passionate female Arabic football commentator. When shouting "هدف", stretch and elongate the word with extreme stadium excitement ("هــــــــدف!") like "Gooooooal!". Speak ONLY the Arabic line.',
        2,
        'AR_FEMALE'
      );
    }
    if (esJobs.length > 0) {
      runQueue(
        esJobs,
        'Puck',
        'You are an explosive native Spanish football commentator. Perform every line with authentic Latin American / Spanish stadium passion. Speak ONLY the Spanish line.',
        3,
        'ES'
      );
    }

    await Promise.all(workers);
    // Clean up marker files
    for (const d of ['audio/ar/male', 'audio/ar/female']) {
      for (const f of fs.readdirSync(d)) {
        if (f.endsWith('.fixed_hadaf')) fs.unlinkSync(path.join(d, f));
      }
    }
    console.log('AR-ES batch complete!');
    return;
  }

  if (mode === 'fr-clean') {
    // Remove old French mp3/wav so all 75 are freshly regenerated cleanly
    for (const f of fs.readdirSync('audio/fr')) {
      if (f.endsWith('.mp3') || f.endsWith('.wav') || f.endsWith('.pcm')) {
        fs.unlinkSync(path.join('audio/fr', f));
      }
    }
    console.log('Cleared old French audio files to regenerate all 75 cleanly.');
    return;
  }

  if (mode === 'fr-batch') {
    const frJobs = [];
    for (const item of v2) {
      const frBase = path.resolve('audio/fr', item.id);
      if (!fs.existsSync(`${frBase}.mp3`) || fs.statSync(`${frBase}.mp3`).size < 2000) {
        const { line, emoDesc } = parseSpokenAndEmotion(item.phrase_fr);
        const extraNote = item.event === 'GOAL'
          ? 'VERY IMPORTANT: Pronounce "But" cleanly and naturally in French with extreme excitement and triumphant energy!'
          : '';
        frJobs.push({
          id: item.id,
          baseOutPath: frBase,
          prompt: `Acting direction: (${emoDesc}). ${extraNote}\nRead aloud ONLY this French script line with authentic, passionate French live football commentator emotion (do NOT speak the acting direction): "${line}"`,
        });
      }
    }

    const batch = frJobs.slice(0, maxBatch);
    console.log(`French remaining: ${frJobs.length}, processing batch of ${batch.length}...`);

    const NUM_WORKERS = 5;
    let doneCount = 0;
    const totalBatch = batch.length;

    const workers = Array.from({ length: NUM_WORKERS }, async () => {
      const worker = new LiveVoiceWorker(
        'Kore',
        'You are a world-class native French football commentator. Perform every line in natural, expressive French with stadium broadcast energy. Pronounce "But" clearly and excitedly. Speak ONLY the French script line.'
      );
      try {
        while (batch.length > 0) {
          const job = batch.shift();
          if (!job) break;
          const pcm = await worker.synthesize(job.prompt);
          await savePcmAsWavAndMp3(pcm, job.baseOutPath);
          doneCount++;
          console.log(`[FR] ${job.id}.mp3 generated (${doneCount}/${totalBatch})`);
        }
      } finally {
        worker.close();
      }
    });

    await Promise.all(workers);
    console.log(`French batch complete! (${doneCount}/${totalBatch})`);
    return;
  }

  // Update per-language manifests and print audit
  const arManifest = v2.map((i) => ({
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

  const frManifest = v2.map((i) => ({
    id: i.id,
    event: i.event,
    phrase: i.phrase_fr,
    audio_url: i.audio_fr,
    audio_full_url: i.audio_full_url_fr,
    wav_url: i.audio_fr.replace('.mp3', '.wav'),
  }));
  fs.writeFileSync('audio/fr/library.json', JSON.stringify(frManifest, null, 2) + '\n');

  const esManifest = v2.map((i) => ({
    id: i.id,
    event: i.event,
    phrase: i.phrase_es,
    audio_url: i.audio_es,
    audio_full_url: i.audio_full_url_es,
    wav_url: i.audio_es.replace('.mp3', '.wav'),
  }));
  fs.writeFileSync('audio/es/library.json', JSON.stringify(esManifest, null, 2) + '\n');

  for (const d of ['audio/ar', 'audio/ar/male', 'audio/ar/female', 'audio/fr', 'audio/es', 'audio/en/male', 'audio/en/female']) {
    const mp3s = fs.readdirSync(d).filter((f) => f.endsWith('.mp3')).length;
    const wavs = fs.readdirSync(d).filter((f) => f.endsWith('.wav')).length;
    console.log(`${d.padEnd(18)}: ${mp3s}/75 MP3s | ${wavs}/75 WAVs`);
  }
}

run().catch((err) => {
  console.error('Error in batch-runner:', err);
  process.exit(1);
});
