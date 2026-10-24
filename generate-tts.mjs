import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { GoogleGenAI, Modality } from '@google/genai';

const execFileAsync = promisify(execFile);

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error('Missing GEMINI_API_KEY environment variable');
  process.exit(1);
}

const ai = new GoogleGenAI({
  apiKey: API_KEY,
  httpOptions: {
    headers: { 'User-Agent': 'aistudio-build' },
  },
});

const EMOTION_DESCRIPTIONS = {
  excited: 'with high-energy stadium excitement and enthusiasm',
  cheerful: 'with a warm, cheerful, upbeat broadcast smile in your voice',
  energetic: 'with fast-paced, punchy, energetic live sports commentary',
  enthusiastic: 'with passionate, welcoming stadium enthusiasm',
  shouting: 'shouting loudly at the top of your lungs like a commentator calling a huge goal',
  screaming: 'screaming with pure football ecstasy and thrilling disbelief as the stadium erupts',
  laughs: 'laughing naturally with joyful disbelief and amazement as you speak',
  breathless: 'taking an audible breathless gasp of adrenaline and speaking breathlessly with excitement',
  gasp: 'giving an audible sharp gasp of shock and awe right before speaking with excitement',
  serious: 'in a serious, firm, authoritative referee-decision commentator tone',
  annoyed: 'in an annoyed, wincing tone reacting to a rough challenge',
  calm: 'in a calm, composed, professional broadcast tone',
  surprised: 'in a surprised, startled tone reacting to a painful tackle',
  disapproving: 'in a disapproving, stern commentator tone',
  tense: 'in a tense, hushed, edge-of-the-seat suspenseful tone',
  whispering: 'whispering in dramatic, hushed stadium suspense',
  dramatic: 'with dramatic, high-stakes theatrical broadcast delivery',
  focused: 'with focused, sharp play-by-play tactical intensity',
  intense: 'with intense, gripping, urgent play-by-play energy',
  amazed: 'with amazed, awestruck wonder at world-class skill',
  impressed: 'with impressed, admiring vocal warmth and praise',
  urgent: 'with urgent, rapid-fire counter-attack speed and intensity',
  disappointed: 'with a disappointed, deflated commentator tone',
  groans: 'giving a brief frustrated groan/sigh before speaking in a disappointed tone',
  'matter-of-fact': 'in a matter-of-fact, concise professional tone',
  relaxed: 'in a relaxed, smooth half-time broadcast tone',
  thoughtful: 'in a thoughtful, reflective, pondering commentator tone',
  emotional: 'with deep, heartfelt emotion at the end of a great match',
};

function buildActingPrompt(phrase) {
  const regex = /\[([^\]]+)\]\s*([^\[]*)/g;
  const directions = [];
  const spokenSegments = [];
  let match;

  while ((match = regex.exec(phrase)) !== null) {
    const tag = match[1].trim().toLowerCase();
    let text = match[2].trim();
    if (!text) continue;

    const desc = EMOTION_DESCRIPTIONS[tag] || `in a ${tag} tone`;
    directions.push(`deliver "${text}" ${desc}`);

    if (tag === 'laughs' && !/^(ha|he|oh)/i.test(text)) {
      text = `Ha ha! ${text}`;
    } else if (tag === 'groans' && !/^(ah|oh|ugh)/i.test(text)) {
      text = `Ahh, ${text}`;
    }
    spokenSegments.push(text);
  }

  const cleanLine =
    spokenSegments.length > 0
      ? spokenSegments.join(' ')
      : phrase.replace(/\[[^\]]+\]/g, '').trim();

  const actingNote =
    directions.length > 0
      ? directions.join('; then ')
      : 'with natural live football commentator energy';

  return `Acting direction: (${actingNote}).\nRead aloud ONLY this exact script line with that exact vocal emotion (do NOT speak the acting direction): "${cleanLine}"`;
}

async function savePcmAsWavAndMp3(pcmBuffer, baseOutPath) {
  const rawPath = `${baseOutPath}.pcm`;
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
  constructor(voiceName) {
    this.voiceName = voiceName;
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
          parts: [
            {
              text: 'You are a world-class live stadium football commentator recording lines in a studio. For each prompt, perform aloud ONLY the quoted script line using the exact vocal emotion, shouting, screaming, laughing, gasping, breathlessness, whispering, or groan described in the acting direction. Sound 100% human, passionate, and authentic. NEVER read the acting directions out loud, and NEVER add extra words before or after the script line.',
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
            fn(new Error('Live session closed unexpectedly'));
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

      if (!pcmBuffer || pcmBuffer.length < 4800) {
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
      try {
        this.session.close();
      } catch {}
      this.session = null;
    }
  }
}

async function main() {
  const libraryPath = path.resolve('library.json');
  const rawLibrary = JSON.parse(fs.readFileSync(libraryPath, 'utf8'));

  const maleDir = path.resolve('en/male');
  const femaleDir = path.resolve('en/female');
  fs.mkdirSync(maleDir, { recursive: true });
  fs.mkdirSync(femaleDir, { recursive: true });

  // 1 male voice (Fenrir) and 3 female voices (Kore, Aoede, Zephyr)
  const MALE_VOICE = 'Fenrir';
  const FEMALE_VOICES = ['Kore', 'Aoede', 'Zephyr'];

  const eventCounters = {};
  const jobs = [];

  for (let i = 0; i < rawLibrary.length; i++) {
    const item = rawLibrary[i];
    const event = item.event.toUpperCase();
    eventCounters[event] = (eventCounters[event] || 0) + 1;
    const idx = eventCounters[event];
    const fileStem = `${event}_${idx}`;
    const femaleVoice = FEMALE_VOICES[(idx - 1) % FEMALE_VOICES.length];
    const prompt = buildActingPrompt(item.phrase);

    item.audio_url = `en/male/${fileStem}.mp3`;
    item.audio_url_male = `en/male/${fileStem}.mp3`;
    item.audio_url_female = `en/female/${fileStem}.mp3`;
    item.voice_male = MALE_VOICE;
    item.voice_female = femaleVoice;

    const maleBase = path.join(maleDir, fileStem);
    const femaleBase = path.join(femaleDir, fileStem);

    if (!fs.existsSync(`${maleBase}.mp3`) || fs.statSync(`${maleBase}.mp3`).size < 1000) {
      jobs.push({
        voice: MALE_VOICE,
        baseOutPath: maleBase,
        prompt,
        label: `male/${fileStem}`,
      });
    }

    if (!fs.existsSync(`${femaleBase}.mp3`) || fs.statSync(`${femaleBase}.mp3`).size < 1000) {
      jobs.push({
        voice: femaleVoice,
        baseOutPath: femaleBase,
        prompt,
        label: `female/${fileStem}`,
      });
    }
  }

  console.log(`Total audio clips remaining to generate: ${jobs.length}`);

  // Group jobs by voice and spawn parallel Live workers per voice
  const jobsByVoice = {};
  for (const job of jobs) {
    if (!jobsByVoice[job.voice]) jobsByVoice[job.voice] = [];
    jobsByVoice[job.voice].push(job);
  }

  let completed = 0;
  const totalJobs = jobs.length;

  const workerPromises = [];
  for (const [voice, voiceJobs] of Object.entries(jobsByVoice)) {
    // Use 4 workers for Male (75 clips) and 2 workers for each of the 3 Female voices (25 clips each) = 10 concurrent sessions
    const numWorkers = voice === MALE_VOICE ? 4 : 2;
    for (let w = 0; w < numWorkers; w++) {
      workerPromises.push(
        (async () => {
          const worker = new LiveVoiceWorker(voice);
          try {
            while (voiceJobs.length > 0) {
              const job = voiceJobs.shift();
              if (!job) break;
              const pcm = await worker.synthesize(job.prompt);
              await savePcmAsWavAndMp3(pcm, job.baseOutPath);
              completed++;
              if (completed % 10 === 0 || completed === totalJobs) {
                console.log(`Progress: ${completed}/${totalJobs} audio clips generated`);
              }
            }
          } finally {
            worker.close();
          }
        })()
      );
    }
  }

  await Promise.all(workerPromises);

  // Write updated manifests
  fs.writeFileSync(libraryPath, JSON.stringify(rawLibrary, null, 2) + '\n');

  const maleLibrary = rawLibrary.map((entry) => ({
    event: entry.event,
    phrase: entry.phrase,
    voice: entry.voice_male,
    audio_url: entry.audio_url_male,
    wav_url: entry.audio_url_male.replace(/\.mp3$/, '.wav'),
  }));
  fs.writeFileSync(
    path.join(maleDir, 'library.json'),
    JSON.stringify(maleLibrary, null, 2) + '\n'
  );

  const femaleLibrary = rawLibrary.map((entry) => ({
    event: entry.event,
    phrase: entry.phrase,
    voice: entry.voice_female,
    audio_url: entry.audio_url_female,
    wav_url: entry.audio_url_female.replace(/\.mp3$/, '.wav'),
  }));
  fs.writeFileSync(
    path.join(femaleDir, 'library.json'),
    JSON.stringify(femaleLibrary, null, 2) + '\n'
  );

  console.log('Successfully generated all male and female emotional TTS audio files!');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
