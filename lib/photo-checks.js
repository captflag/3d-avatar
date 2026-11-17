/**
 * The photo checks the studio and the app both run before a photo is built
 * from: MediaPipe's face landmarker in the browser (the same model the
 * pipeline reads photos with), and what each check needs (compose's limits,
 * where the numbers are). Moved out of studio.html so the two cannot drift.
 */
export const FACE_PX = 500;             // cheek to cheek, as compose.photo_advice measures; twice its GOOD_FACE_PX
export const MAX_TURN = 10;             // degrees left/right or up/down, front photo
export const MAX_ROLL = 8;              // degrees of head tilt
export const MAX_SMILE = 0.3;           // MediaPipe score where the shape fit starts damping a smile
export const MAX_BLINK = 0.4;
export const MAX_LIGHT_SPLIT = 0.25;    // cheek against cheek, share of the brighter
export const SHARP_PX = 320;            // the face is scaled to this width before its sharpness is measured
export const MIN_SHARP = 0.4;           // sharpness() below it is blurred; see side-session-notes/guided-capture.md
// The cheeks' brightness (0-255) below which the photo is too dark, and the eye
// openings' brightest tenth, as a share of the cheeks', below which the eyes look
// covered (dark glasses): compose.MIN_CHEEK and MIN_EYES, where the numbers are.
export const MIN_CHEEK = 70;
export const MIN_EYES = 0.35;
export const STEADY_FRAMES = 15;
// Turned 30-45 degrees. At 11-23 real turned photos barely moved the depth;
// on a rendered test head two shots at 50 degrees left a third of the depth
// error two at 35 did. The cap is for the landmarks: at 55 the far eye hides
// and MediaPipe's reading wanders (it read the two sides 52 and 41). MediaPipe
// reads true turns of 20, 35 and 50 degrees as about 20, 29 and 42, and finds
// no face side-on: these are about its readings for 30 and 46.
export const SIDE_TURN = [25, 40];
// MediaPipe's yaw sign when the head turns to the person's own left.
export const LEFT = 1;
export const VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
export const landmarkers = {};

/** One detector per mode: VIDEO tracks a face from frame to frame, which
 *  unrelated stills (checkStill) would confuse. */
export async function getLandmarker(mode = 'VIDEO') {
  if (landmarkers[mode]) return landmarkers[mode];
  const { FilesetResolver, FaceLandmarker } = await import(`${VISION}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
  landmarkers[mode] = await FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
    runningMode: mode, numFaces: 2,
    outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  });
  return landmarkers[mode];
}

export const sampler = document.createElement('canvas');
/** Mean brightness round a point, in a small copy of the frame. */
export function brightness(source, width, height, x, y, radius) {
  const scale = 320 / width;
  sampler.width = 320; sampler.height = Math.round(height * scale);
  const ctx = sampler.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, sampler.width, sampler.height);
  const r = Math.max(2, Math.round(radius * scale));
  const data = ctx.getImageData(Math.round(x * scale) - r, Math.round(y * scale) - r, 2 * r, 2 * r).data;
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  return sum / (data.length / 4);
}

/** The brightest tenth of the box [x0, y0, x1, y1] (an eye opening), in a small
 *  copy of the frame. */
export function brightTenth(source, width, height, x0, y0, x1, y1) {
  const scale = 320 / width;
  sampler.width = 320; sampler.height = Math.round(height * scale);
  const ctx = sampler.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, sampler.width, sampler.height);
  const bx = Math.round(x0 * scale), by = Math.round(y0 * scale);
  const data = ctx.getImageData(bx, by, Math.max(1, Math.round(x1 * scale) - bx), Math.max(1, Math.round(y1 * scale) - by)).data;
  const lum = [];
  for (let i = 0; i < data.length; i += 4) lum.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  lum.sort((a, b) => a - b);
  return lum[Math.floor(0.9 * (lum.length - 1))];
}

/** How sharp the face in the box [x0, y0, x1, y1] is, 0 (flat) to 1: the share
 *  of its edges' contrast that blurring it further takes away (Crete-Roffet's
 *  re-blur measure, per axis, the blurrier axis). A blurred face has little left
 *  to lose; unlike plain edge strength this does not depend on how much texture
 *  or contrast the face has. Scaled to SHARP_PX wide so any photo size compares:
 *  a small face enlarged to that width reads soft, as it is to the pipeline. */
export function sharpness(source, x0, y0, x1, y1) {
  const scale = SHARP_PX / (x1 - x0);
  const w = SHARP_PX, h = Math.max(12, Math.round((y1 - y0) * scale));
  sampler.width = w; sampler.height = h;
  const ctx = sampler.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, x0, y0, x1 - x0, y1 - y0, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2];
  const R = 4;   // a 9-pixel box blur
  const kept = (step, along, across) => {
    let edges = 0, lost = 0;
    for (let a = 0; a < across; a++) for (let t = R + 1; t < along - R; t++) {
      const at = (u) => g[a * (step === 1 ? w : 1) + u * step];
      let box = 0, prev = 0;
      for (let k = -R; k <= R; k++) { box += at(t + k); prev += at(t - 1 + k); }
      const d = Math.abs(at(t) - at(t - 1)), db = Math.abs(box - prev) / (2 * R + 1);
      edges += d; lost += Math.max(0, d - db);
    }
    return edges ? lost / edges : 0;
  };
  return Math.min(kept(1, w, h), kept(w, h, w));
}

/** Each check on one frame (`found`: the detector's result for it): [{ ok, text }],
 *  with the measures behind them on the array (`measured`). `side`: null for
 *  the front photo, else the yaw sign wanted (LEFT or -LEFT). */
export function assess(found, source, width, height, side = null) {
  const faces = found.faceLandmarks || [];
  const out = [];
  const add = (ok, good, bad) => out.push({ ok, text: ok ? good : bad });
  if (faces.length !== 1) {
    add(false, '', faces.length ? 'Only one face in the picture, please' : 'No face found: look at the camera');
    return out;
  }
  const p = faces[0];
  const at = (i) => [p[i].x * width, p[i].y * height];
  const faceWidth = Math.abs(at(454)[0] - at(234)[0]);
  // Turned, the face shows narrower across the cheeks.
  const needPx = side === null ? FACE_PX : Math.round(0.8 * FACE_PX);
  add(faceWidth >= needPx, 'Face close enough',
      `Come closer: face ${Math.round(faceWidth)} px across, needs ${needPx}`);

  // Head pose from the face's transform (column-major 4x4).
  const m = found.facialTransformationMatrixes[0].data;
  const deg = 180 / Math.PI;
  const pitch = Math.atan2(m[6], m[10]) * deg, yaw = Math.asin(-m[2]) * deg, roll = Math.atan2(m[1], m[0]) * deg;
  if (side === null) {
    add(Math.abs(yaw) <= MAX_TURN && Math.abs(pitch) <= MAX_TURN, 'Facing the camera',
        Math.abs(yaw) > MAX_TURN ? 'Turn to face the camera' : 'Keep your chin level');
  } else {
    const [least, most] = SIDE_TURN;
    const turn = yaw * side;   // positive: the way wanted
    add(turn >= least && turn <= most && Math.abs(pitch) <= MAX_TURN, `Turned ${Math.round(turn)}° the right way`,
        turn < -MAX_TURN ? `Wrong way: turn to your ${side === LEFT ? 'left' : 'right'}`
          : turn < least ? `Turn further: ${Math.round(Math.max(turn, 0))}°, needs ${least}`
          : turn > most ? `Not quite so far: ${Math.round(turn)}°, at most ${most}` : 'Keep your chin level');
  }
  add(Math.abs(roll) <= MAX_ROLL, 'Head upright', 'Straighten your head');

  const score = Object.fromEntries(found.faceBlendshapes[0].categories.map((c) => [c.categoryName, c.score]));
  add(Math.max(score.mouthSmileLeft, score.mouthSmileRight, score.jawOpen) <= MAX_SMILE, 'Relaxed, mouth closed',
      'Relax your face: mouth closed, no smile');
  add(Math.max(score.eyeBlinkLeft, score.eyeBlinkRight) <= MAX_BLINK, 'Eyes open', 'Open your eyes');

  const ys = [10, 152].map((i) => at(i)[1]);
  const sharp = sharpness(source, Math.min(at(234)[0], at(454)[0]), ys[0], Math.max(at(234)[0], at(454)[0]), ys[1]);
  add(sharp >= MIN_SHARP, 'Sharp', 'Blurred: hold still, or more light');
  out.measured = { faceWidth: Math.round(faceWidth), yaw: +yaw.toFixed(1), pitch: +pitch.toFixed(1),
                   roll: +roll.toFixed(1), sharp: +sharp.toFixed(2),
                   mouth: +Math.max(score.mouthSmileLeft, score.mouthSmileRight, score.jawOpen).toFixed(2) };

  // A turned face is lit unevenly by its turn, and the hair was checked on the front photo.
  if (side !== null) return out;

  const radius = 0.06 * faceWidth;
  const left = brightness(source, width, height, ...at(50), radius);
  const right = brightness(source, width, height, ...at(280), radius);
  out.measured.lightSplit = +(Math.abs(left - right) / Math.max(left, right, 1)).toFixed(2);
  add(out.measured.lightSplit <= MAX_LIGHT_SPLIT, 'Even light',
      'The light falls on one side: face the window or lamp');
  out.measured.cheek = Math.round((left + right) / 2);
  add(out.measured.cheek >= MIN_CHEEK, 'Light enough', 'Too dark: more light on your face');
  const eye = (a, b, c, d) => brightTenth(source, width, height, Math.min(at(a)[0], at(b)[0]), at(c)[1],
                                          Math.max(at(a)[0], at(b)[0]), at(d)[1]);
  out.measured.eyes = +(((eye(33, 133, 159, 145) + eye(263, 362, 386, 374)) / 2) / Math.max(out.measured.cheek, 1)).toFixed(2);
  add(out.measured.eyes >= MIN_EYES, 'Eyes uncovered', 'Take off dark glasses');

  const top = p[10].y - 0.45 * (p[152].y - p[10].y);
  add(top > 0 && p[152].y < 0.98, 'Head and hair in the picture',
      'Move back or lower the camera: the top of your hair must be in the picture');
  return out;
}
