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

async function savePcm(pcmBuffer, baseOutPath) {
  const pcmPath = `${baseOutPath}.pcm`;
  fs.writeFileSync(pcmPath, pcmBuffer);
  try {
    await execFileAsync('ffmpeg', [
      '-y', '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', pcmPath, `${baseOutPath}.wav`,
    ]);
    await execFileAsync('ffmpeg', [
      '-y', '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', pcmPath, '-codec:a', 'libmp3lame', '-qscale:a', '2', `${baseOutPath}.mp3`,
    ]);
  } finally {
    if (fs.existsSync(pcmPath)) fs.unlinkSync(pcmPath);
  }
}

async function synthOne(voice, instruction, prompt, baseOutPath) {
  let chunks = [];
  let resolveTurn;
  const session = await ai.live.connect({
    model: 'gemini-3.8-live',
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
      systemInstruction: { parts: [{ text: instruction }] },
    },
    callbacks: {
      onmessage: (msg) => {
        const parts = msg.serverContent?.modelTurn?.parts || [];
        for (const p of parts) {
          if (p.inlineData?.data) chunks.push(Buffer.from(p.inlineData.data, 'base64'));
        }
        if (msg.serverContent?.turnComplete && resolveTurn) {
          resolveTurn(Buffer.concat(chunks));
        }
      },
    },
  });

  const pcm = await new Promise((res) => {
    resolveTurn = res;
    session.sendClientContent({
      turns: [{ role: 'user', parts: [{ text: prompt }] }],
      turnComplete: true,
    });
  });

  try { session.close(); } catch {}
  await savePcm(pcm, baseOutPath);
  console.log(`Generated: ${baseOutPath}.mp3 (${pcm.length} bytes)`);
}

async function main() {
  const v2 = JSON.parse(fs.readFileSync('library_v2.json', 'utf8'));

  // 1. Spanish INTRO_1
  const esIntro1 = v2.find(i => i.id === 'INTRO_1');
  if (esIntro1 && !fs.existsSync('audio/es/INTRO_1.mp3')) {
    const text = "¡Hola a todos y muy bienvenidos! ¡Los equipos ya están listos y el partido está a punto de comenzar!";
    await synthOne(
      'Puck',
      'You are a high-energy Spanish football commentator. Speak with authentic stadium excitement.',
      `Acting direction: (excited). Read aloud ONLY this Spanish script line: "${text}"`,
      'audio/es/INTRO_1'
    );
  }

  // 2. French missing 8
  const missingFrIds = ['INTRO_3', 'HALFTIME_1', 'HALFTIME_2', 'HALFTIME_3', 'HALFTIME_4', 'HALFTIME_5', 'FULLTIME_1', 'FULLTIME_2'];
  for (const id of missingFrIds) {
    const outPath = `audio/fr/${id}`;
    if (fs.existsSync(`${outPath}.mp3`)) continue;
    const item = v2.find(i => i.id === id);
    if (!item) continue;
    const cleanLine = item.phrase_fr.replace(/\[[^\]]+\]/g, '').trim().replace(/\bbu+t+\b/gi, 'But !');
    await synthOne(
      'Kore',
      'You are a dynamic French football commentator in a recording booth. NEVER say stage directions.',
      `Acting direction: (passionate broadcast delivery). Read aloud ONLY this French script line: "${cleanLine}"`,
      outPath
    );
  }

  console.log('All remaining 9 audio files generated successfully!');
}

main().catch(err => {
  console.error('Fatal in finish-last-9:', err);
  process.exit(1);
});
