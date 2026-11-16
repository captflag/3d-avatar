/**
 * Shared lighting for the viewer and the thumbnail/PNG renderer.
 *
 * Physically-based materials need something to reflect. Lit with a flat ambient
 * light and one weak directional, as this was, every surface returns the same
 * value from every angle and the avatar reads as clay. An environment map gives
 * the materials a room to reflect, a key gives shape, a rim separates the figure
 * from the background, and a contact shadow stops it hovering.
 *
 * This does not make a low-detail avatar photoreal. It removes the reasons a
 * detailed one would still look fake.
 */
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

/**
 * Ambient occlusion: the soft darkening where surfaces crowd each other —
 * under the chin, inside the collar, where an arm meets the body, in the folds
 * of cloth and the corners of the eyes. Without it a figure is lit from inside
 * and every part floats against the next. Ground-truth AO (GTAO) is screen
 * space, so it costs one extra pass and nothing in the exported file.
 *
 * `radius` is in scene units; the avatars are about 17 units tall, so 0.6 is
 * about 6 cm: creases and contacts, not whole-body shading. Returns
 * { render(), setSize(w, h) } to use in place of renderer.render().
 *
 * `coverage` (a render target from renderCoverage) replaces the alpha of a
 * transparent export with the figure's real coverage.
 */
/** A repeatable stand-in for Math.random (mulberry32). */
function seeded(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function withOcclusion(renderer, scene, camera, { radius = 0.6, intensity = 0.9, coverage = null } = {}) {
  const size = renderer.getSize(new THREE.Vector2());
  // Multisampled: rendering through the composer bypasses the canvas's own
  // antialiasing, so without samples every edge stairsteps — and the hair's
  // alpha-to-coverage edge (applyMaterials) needs samples to soften at all.
  const pixelRatio = renderer.getPixelRatio();
  const target = new THREE.WebGLRenderTarget(size.x * pixelRatio, size.y * pixelRatio,
    { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, target);
  composer.addPass(new RenderPass(scene, camera));
  // Built under a fixed seed: GTAOPass makes its denoiser's noise with three's
  // SimplexNoise, which draws from Math.random, so every render's occlusion
  // came out speckled differently. Two renders of one GLB differed at 48,000
  // pixels, by up to 94 levels, and ref-man's identity score moved 0.681-0.686.
  const random = Math.random;
  Math.random = seeded(1);
  let ao;
  try { ao = new GTAOPass(scene, camera, size.x, size.y); } finally { Math.random = random; }
  ao.updateGtaoMaterial({ radius, distanceExponent: 1.5, thickness: 1.0, scale: 1.0, samples: 16 });
  ao.blendIntensity = intensity;
  composer.addPass(ao);
  composer.addPass(new OutputPass());       // tone mapping and sRGB, as a plain render would
  if (coverage) {
    const alpha = new ShaderPass({
      uniforms: { tDiffuse: { value: null }, tCoverage: { value: null } },
      vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: `uniform sampler2D tDiffuse; uniform sampler2D tCoverage; varying vec2 vUv;
        void main() { vec4 c = texture2D(tDiffuse, vUv); gl_FragColor = vec4(c.rgb, max(c.a, texture2D(tCoverage, vUv).r)); }`,
    });
    // Set after construction: ShaderPass clones its uniforms, and a clone of
    // a render target's texture is an empty texture.
    alpha.uniforms.tCoverage.value = coverage.texture;
    composer.addPass(alpha);
  }
  return {
    render: () => composer.render(),
    setSize: (w, h) => composer.setSize(w, h),
    dispose: () => composer.dispose(),
  };
}

/**
 * How much of each pixel the figure covers, for the alpha of a transparent
 * export. Alpha-to-coverage leaves each covered sample with the fragment's own
 * alpha, so where a cut-out card (hair, lashes, the beard shell) lies over
 * opaque skin the export came out part transparent: pale specks and outlines
 * on a white page. Drawn again in plain white, the resolved samples are the
 * true coverage. Only `root` is drawn; the ground's shadow keeps its own alpha.
 * The caller disposes the returned render target.
 */
export function renderCoverage(renderer, scene, camera, root) {
  const size = renderer.getSize(new THREE.Vector2()).multiplyScalar(renderer.getPixelRatio());
  const target = new THREE.WebGLRenderTarget(size.x, size.y, { samples: 4 });
  const mine = new Set();
  root.traverse((o) => { if (o.isMesh) mine.add(o); });
  const white = (m) => {
    const w = m.clone();
    w.onBeforeCompile = (shader) => {
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <dithering_fragment>', '#include <dithering_fragment>\n\tgl_FragColor.rgb = vec3(1.0);');
    };
    w.customProgramCacheKey = () => 'coverage';
    return w;
  };
  const restore = [];
  scene.traverse((o) => {
    if (!o.isMesh) return;
    if (mine.has(o)) {
      const original = o.material;
      o.material = Array.isArray(original) ? original.map(white) : white(original);
      restore.push(() => { [].concat(o.material).forEach((m) => m.dispose()); o.material = original; });
    } else if (o.visible) {
      o.visible = false;
      restore.push(() => { o.visible = true; });
    }
  });
  const clearColour = renderer.getClearColor(new THREE.Color());
  const clearAlpha = renderer.getClearAlpha();
  renderer.setRenderTarget(target);
  renderer.setClearColor(0x000000, 0);
  renderer.clear();
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  renderer.setClearColor(clearColour, clearAlpha);
  restore.forEach((undo) => undo());
  return target;
}

// Exposure 1.0 (was 1.05) pays for the side lights: the cheeks get the same
// light as before them, so delight.rendered's model of the studio holds.
export function setupRealism(renderer, scene, { exposure = 1.0 } = {}) {
  // Khronos PBR Neutral: keeps the base colour's hue and saturation up to the
  // highlights. ACES halved the skin's chroma (forehead a*/b* 7.8/13.5 in the
  // texture, 3.5/6.8 on screen) and AgX more so; measured against the photos
  // over forehead, cheeks and nose, skin was dE 14 off under ACES and 5 under
  // Neutral. Dimmer lights did not help: they only darkened every curve.
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = exposure;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  // The room's lights tipped 30 degrees towards the face, a portrait setup.
  // Lit straight from the ceiling, brow ridge and eye sockets faced away from
  // most of the room's light while the forehead faced into it, so the shading
  // the photo already carries there came out twice over: under-brow skin 38
  // L* darker than the forehead against the photo's 16, light brows 24 against
  // 18. Tipped further (45-75) it matched those better still but flattened
  // the nose and lips into a pasted-on look.
  // And turned 14 degrees about the vertical: the room's light is lopsided,
  // and on a flat grey head with no other lights one cheek rendered 39 L*
  // brighter than the other (+70 at -20 degrees, -15 at +20; zero near +14).
  // On the photo-textured heads that read as a lit cheek and a shadowed one
  // whatever the photo showed.
  // Tipped 37 rather than 30 since the side lights below: with them, on the
  // five test faces at the photo's head pose, render minus photo L* summed
  // over forehead, cheeks and chin went 169 -> 134 (level 97.5 -> 97), jaw
  // 103 -> 90; the left-right cheek split stayed where it was.
  scene.environmentRotation.set(THREE.MathUtils.degToRad(37), THREE.MathUtils.degToRad(14), 0);

  // Key: front and to one side, the light that models the face and casts the
  // shadow under the chin and between the arms.
  // Softer than it was (1.7): with a weak fill it cut faces in half, harder
  // than any portrait lighting.
  const key = new THREE.DirectionalLight(0xfff2e6, 1.35);
  key.position.set(1.6, 2.4, 2.6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 4;
  scene.add(key);
  scene.add(key.target);

  // Fill: from the other side and in front, about half the key, so the
  // shadow side of a face keeps its form instead of going flat grey. At 0.4
  // it was a quarter of the key: harsh.
  const fill = new THREE.DirectionalLight(0xe8eeff, 0.75);
  fill.position.set(-2.0, 0.8, 2.0);
  scene.add(fill);

  // Wrap: light from the whole sky above and a warm bounce from below, the
  // soft all-round light of a studio that keeps skin from looking hard-lit.
  scene.add(new THREE.HemisphereLight(0xfff6ee, 0x8a7060, 0.45));

  // Rim: behind and above. This is the bright outline on shoulders, hair and
  // jaw in studio portraits, and the main thing that lifts a figure off a dark
  // background instead of letting it sink into it.
  // Lower than it was (1.4): stronger, it lit ears and jaw from behind as a
  // cold white edge.
  const rim = new THREE.DirectionalLight(0xffffff, 1.0);
  rim.position.set(-0.6, 2.2, -3.0);
  scene.add(rim);

  // Side lights: one each side, level and 10 degrees behind the face, the
  // wrap of a big soft source. Without them the face fell off too fast away
  // from the front: on the five test faces the cheeks (which face partly
  // sideways) and the jaw got 0.6-0.8 of the light that would render their
  // photo's brightness while the forehead got 1.2. Photos are lit softer
  // than the studio was, not from lower: delight's fit puts their light 45-75
  // degrees up, the studio's 15-20. Render minus photo L* summed over
  // forehead, cheeks and chin, level 139 -> 97.5, at the photo's head pose
  // 199.5 -> 169, jaw 321 -> 185. From 80 degrees (in front of the ear) less
  // so, from 60 less again; from above (30 up) it lit foreheads, not cheeks.
  const wraps = [1, -1].map((side) => {
    const wrap = new THREE.DirectionalLight(0xffffff, 2.0);
    wrap.position.set(side * 3.0 * Math.sin(THREE.MathUtils.degToRad(100)), 0, 3.0 * Math.cos(THREE.MathUtils.degToRad(100)));
    scene.add(wrap);
    return wrap;
  });

  return { pmrem, key, fill, rim, wraps };
}

/**
 * The stylized style's portrait (the reference cartoon avatar): a plain warm
 * backdrop, a key from front-left and above that models the face, a fill at
 * `fill` of it, a warm rim from behind right, and the side lights (which
 * flatten a face to match a photo's soft light) turned down. After
 * groundAndShadow, which places the key for the realistic style.
 */
export const PORTRAIT = {
  backdrop: [222, 165, 130],                 // sRGB
  key: { azimuth: -35, elevation: 30, intensity: 1.6, colour: 0xfff0e0 },
  fill: 0.45,
  rim: { intensity: 1.2, colour: 0xffd2a8 },
  wrap: 0.6,
};
export function portraitLights(scene, lights, model) {
  const P = PORTRAIT, reach = 1.03 * Math.max(...visibleBox(model).getSize(new THREE.Vector3()).toArray());
  const az = THREE.MathUtils.degToRad(P.key.azimuth), el = THREE.MathUtils.degToRad(P.key.elevation);
  lights.key.position.set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)).multiplyScalar(reach);
  lights.key.intensity = P.key.intensity;
  lights.key.color.set(P.key.colour);
  lights.fill.position.set(2.0, 0.8, 2.0);
  lights.fill.intensity = P.fill * P.key.intensity;
  lights.fill.color.set(0xfff4ea);
  lights.rim.position.set(1.2, 2.0, -3.0);
  lights.rim.intensity = P.rim.intensity;
  lights.rim.color.set(P.rim.colour);
  for (const w of lights.wraps) w.intensity = P.wrap;
  scene.background = new THREE.Color().setRGB(...P.backdrop.map((v) => v / 255), THREE.SRGBColorSpace);
}

/**
 * Fit the key light's shadow camera round the model and put a ground under it.
 *
 * The ground uses ShadowMaterial, which draws only the shadow and is otherwise
 * transparent, so a PNG export still has a clean alpha background but the feet
 * now touch something.
 */
/**
 * The model's bounds over what is drawn. Box3.setFromObject counts hidden
 * meshes too, and every file carries every haircut and pair of glasses
 * hidden: a tall cut added to the catalogue moved the camera and the key
 * light's shadow for every avatar, and their scores with them.
 */
export function visibleBox(model) {
  model.updateMatrixWorld(true);
  const box = new THREE.Box3();
  model.traverseVisible((o) => { if (o.isMesh) box.expandByObject(o); });
  return box;
}

export function groundAndShadow(scene, model, key) {
  const box = visibleBox(model);
  const size = box.getSize(new THREE.Vector3());
  const reach = Math.max(size.x, size.y, size.z);

  model.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });

  const cam = key.shadow.camera;
  cam.left = -reach; cam.right = reach; cam.top = reach; cam.bottom = -reach;
  cam.near = 0.1; cam.far = reach * 6;
  // More frontal than it was (0.35, 0.55, 0.6), the classic portrait key
  // about 30 degrees off the nose: less of the face falls into shadow.
  key.position.set(reach * 0.3, reach * 0.5, reach * 0.85);
  key.target.position.set(0, 0, 0);
  cam.updateProjectionMatrix();

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(reach * 4, reach * 4),
    new THREE.ShadowMaterial({ opacity: 0.32 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = box.min.y;
  ground.receiveShadow = true;
  scene.add(ground);
  return ground;
}

/**
 * Show the face the photo was making. The file keeps it in each mesh's extras
 * (`photoExpression`, ARKit blendshape weights) and leaves its own weights at
 * rest, so a rig driven by face tracking or animation starts neutral; a
 * portrait of the person should not.
 */
export function applyPhotoExpression(root) {
  root.traverse((o) => {
    const expression = o.userData?.photoExpression;
    if (!expression || !o.morphTargetDictionary) return;
    for (const [name, weight] of Object.entries(expression)) {
      const i = o.morphTargetDictionary[name];
      if (i !== undefined) o.morphTargetInfluences[i] = weight;
    }
  });
}

/**
 * Turn the head as it was in the photo (the file's `photoHeadPose`, a world
 * rotation), by the neck. The file's own pose stays level; this is for a 2D
 * render meant to sit beside the photo.
 */
export function applyPhotoPose(root) {
  let pose = null, neck = null;
  root.traverse((o) => {
    if (!pose && o.userData?.photoHeadPose) pose = o.userData.photoHeadPose;
    if (!neck && o.isBone && o.name === 'neck01') neck = o;
  });
  if (!pose || !neck) return;
  root.updateMatrixWorld(true);
  const parent = neck.parent.getWorldQuaternion(new THREE.Quaternion());
  const turn = new THREE.Quaternion(...pose);
  // A world turn, put in the neck's parent's frame.
  neck.quaternion.premultiply(parent.clone().invert().multiply(turn).multiply(parent));
}

/**
 * Display fixes only. Surface finish comes from the file.
 *
 * This used to set roughness per surface here, which made the studio look
 * right while every exported file still said 0.85 for everything, so the
 * avatar looked like plaster in Blender or Unity. The values now live in the
 * GLB, and the viewer shows what any engine would.
 */
export function applyMaterials(root, shading = SHADING) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    const materials = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of materials) {
      if (m && m.map) m.map.colorSpace = THREE.SRGBColorSpace;
      // Alpha-tested cards (hair, brows, lashes) cut each texel to all or
      // nothing, which draws the hair's front edge as a hard, ragged line.
      // Alpha-to-coverage turns the cut into partial coverage per sample.
      if (m && m.alphaTest > 0) m.alphaToCoverage = true;
    }
  });
  shadeSkinAndEyes(root, shading);
  chooseHair(root);
  chooseGlasses(root);
}

/**
 * The cartoon style, in place of applyMaterials: every material a toon one
 * (three bands of light, no highlights), the skin's photo texture flattened
 * to soft areas of colour, and a dark outline round the head, body and
 * clothes (an inverted hull: each mesh drawn again, back faces only, pushed
 * out along its normals). The head wears the file's cartoon paint where it
 * has one (its material's `toonTexture`, compose's cartoon_face, loaded
 * through the glTF `parser`). The same files as the realistic style.
 */
export const TOON = {
  bands: [0.62, 0.84, 1.0], skinTexels: 256, outline: 0.03, outlineColour: 0x2a1d1a,
  // Cartoon proportions, from what the file already has: the lids opened
  // (ARKit eyeWide), the eyeballs a little larger to fill them, a faint
  // smile, and the head larger against the body (its bone's scale).
  shapes: { eyeWideLeft: 1.0, eyeWideRight: 1.0, mouthSmileLeft: 0.15, mouthSmileRight: 0.15 },
  eyeScale: 1.18, headScale: 1.08,
};
let toonGradient = null;
// A bone by its name in the file: three's glTF loader drops the dots
// ("upperarm01.R" loads as "upperarm01R"), so look it up the same way.
const boneNamed = (root, name) => root.getObjectByName(THREE.PropertyBinding.sanitizeNodeName(name));
function flattened(texture, size) {
  const image = texture.image;
  if (!image?.width) return texture;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, 0, 0, size, size);       // shrunk: pores, wrinkles and grain gone
  const out = new THREE.CanvasTexture(canvas);
  out.colorSpace = THREE.SRGBColorSpace;
  out.flipY = texture.flipY;
  out.wrapS = texture.wrapS; out.wrapT = texture.wrapT;
  return out;
}
export function applyToon(root, parser = null) {
  if (!toonGradient) {
    toonGradient = new THREE.DataTexture(new Uint8Array(TOON.bands.map((b) => Math.round(b * 255))), TOON.bands.length, 1, THREE.RedFormat);
    // Blended between bands, not stepped: hard steps broke into patches along
    // the head mesh's seams (the forehead's middle, where head meets neck).
    toonGradient.minFilter = toonGradient.magFilter = THREE.LinearFilter;
    toonGradient.needsUpdate = true;
  }
  const outlined = [], loading = [];
  root.traverse((o) => {
    if (!o.isMesh) return;
    const old = o.material;
    const skin = old.name === 'head' || old.name === 'body';
    if (old.map) old.map.colorSpace = THREE.SRGBColorSpace;
    o.material = new THREE.MeshToonMaterial({
      name: old.name, color: old.color, gradientMap: toonGradient,
      // The body's generated skin to nearly one colour, as the head's paint is.
      map: old.map && skin ? flattened(old.map, old.name === 'body' ? 16 : TOON.skinTexels) : old.map,
      ...(old.userData?.styleTint ? { color: old.color.clone().multiply(new THREE.Color(...old.userData.styleTint)) } : {}),
      alphaTest: old.alphaTest, transparent: old.transparent, opacity: old.opacity, side: old.side,
    });
    if (old.alphaTest > 0) o.material.alphaToCoverage = true;
    const paint = old.userData?.toonTexture;
    if (paint !== undefined && parser) {
      const material = o.material;
      loading.push(parser.getDependency('texture', paint).then((t) => {
        t.colorSpace = THREE.SRGBColorSpace;
        material.map = t;
        material.needsUpdate = true;
      }));
    }
    // Outlined: the solid surfaces. Not cards (hair, brows, lashes), eyes, teeth or glasses.
    if (!(old.alphaTest > 0) && !old.transparent && !/eye|teeth|tongue|glasses|lash|brow/.test(old.name)) outlined.push(o);
  });
  for (const o of outlined) {
    const material = new THREE.MeshBasicMaterial({ color: TOON.outlineColour, side: THREE.BackSide });
    material.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>',
        `#include <begin_vertex>\ntransformed += normalize( normal ) * ${TOON.outline.toFixed(4)};`);
    };
    const hull = o.isSkinnedMesh ? new THREE.SkinnedMesh(o.geometry, material) : new THREE.Mesh(o.geometry, material);
    if (o.isSkinnedMesh) hull.bind(o.skeleton, o.bindMatrix);
    if (o.morphTargetInfluences) hull.morphTargetInfluences = o.morphTargetInfluences;   // shared: expressions follow
    hull.castShadow = false;
    o.add(hull);                                    // a child: hidden with its haircut or garment
  }
  proportion(root, TOON);
  chooseHair(root);
  chooseGlasses(root);
  return Promise.all(loading);                      // the cartoon paint, once loaded
}

// A style's proportions: its face shapes, eyeball and head scale.
function proportion(root, style) {
  root.traverse((o) => {
    if (!o.morphTargetDictionary) return;
    for (const [name, value] of Object.entries(style.shapes)) {
      const i = o.morphTargetDictionary[name];
      if (i !== undefined) o.morphTargetInfluences[i] = value;
    }
  });
  boneNamed(root, 'head')?.scale.setScalar(style.headScale);
  for (const eye of ['eye.L', 'eye.R']) boneNamed(root, eye)?.scale.setScalar(style.eyeScale);
}

/**
 * The stylized style (a game character's look): the realistic style's
 * materials and studio light, the head wearing the file's stylized paint
 * (its material's `styleTexture`: compose's stylized_skin under the
 * cartoon's drawn brows, lips and lashes), and the eyes a little larger.
 */
// Eyes a little wide and the eyeballs larger: morph.STYLIZE already opens the
// lids, and wide eyes (0.6) on top of it showed white all round the iris.
export const STYLIZED = { shapes: { eyeWideLeft: 0.25, eyeWideRight: 0.25, mouthSmileLeft: 0.25, mouthSmileRight: 0.25,
                                    stylize: 1.0 },      // the file's head shape for this style (morph.STYLIZE)
                          eyeScale: 1.2, headScale: 1.0,
                          // The cornea's reflection of the cool studio room over a small dark iris
                          // read as grey-blue eyes (the example photo, iris read dark brown).
                          eyeReflection: 0.3,
                          // A clear coat this strong: the key light's catchlight in each eye.
                          catchlight: 1.0,
                          // A soft broad sheen on forehead and nose, not the realistic
                          // roughness map's tight T-zone shine.
                          skin: { roughness: 0.5, specular: 0.5 } };
// Parts of the stylized style that `off` (a Set, the page's ?off= list) leaves
// out, to measure what each costs in likeness (style_score.py): paint (the
// file's styleTexture and styleTint), skin (STYLIZED_SHADING, smooth and
// broad-sheened skin), eyes (reflection, catchlight, size), expression (wide
// eyes, smile), headshape (the stylize blendshape). The page's own: light, motion.
export const STYLIZED_PARTS = ['paint', 'skin', 'eyes', 'expression', 'headshape', 'light', 'motion'];
export function applyStylized(root, parser = null, off = new Set()) {
  applyMaterials(root, off.has('skin') ? SHADING : STYLIZED_SHADING);
  const loading = [];
  root.traverse((o) => {
    const m = o.material, tint = m?.userData?.styleTint;
    if (o.isMesh && tint && !off.has('paint')) {
      // The body's skin tinted as the face was (compose: appearance.stylized_skin's gain).
      m.color.multiply(new THREE.Color(...tint));
      m.emissive.multiply(new THREE.Color(...tint));
    }
    if (o.isMesh && m.name === 'eyes' && !off.has('eyes')) {
      m.envMapIntensity = STYLIZED.eyeReflection;
      m.clearcoat = STYLIZED.catchlight;
    }
    if (o.isMesh && ['head', 'body'].includes(m.name) && !off.has('skin')) {
      // Painted skin is smooth: no fine relief from the photo's (the file's normal map).
      m.normalMap = null;
      m.roughnessMap = null;
      m.roughness = STYLIZED.skin.roughness;
      m.specularIntensity = STYLIZED.skin.specular;
      m.needsUpdate = true;
    }
  });
  root.traverse((o) => {
    const m = o.material, paint = m?.userData?.styleTexture;
    if (!o.isMesh || paint === undefined || !parser || off.has('paint')) return;
    loading.push(parser.getDependency('texture', paint).then((t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      m.map = t;
      if (m.emissiveMap) m.emissiveMap = t;          // the baked half (SHADING.skin) paints it too
      m.needsUpdate = true;
    }));
  });
  proportion(root, stylizedShapes(off));
  return Promise.all(loading);
}

/** STYLIZED with the parts in `off` left out, for proportion() and idleMotion(). */
export function stylizedShapes(off = new Set()) {
  const shapes = { ...STYLIZED.shapes };
  if (off.has('expression')) for (const k of ['eyeWideLeft', 'eyeWideRight', 'mouthSmileLeft', 'mouthSmileRight']) shapes[k] = 0;
  if (off.has('headshape')) shapes.stylize = 0;
  return { ...STYLIZED, shapes, eyeScale: off.has('eyes') ? 1.0 : STYLIZED.eyeScale };
}

/**
 * Idle life for the cartoon style: blinks every few seconds, breathing in
 * the chest and shoulders, a slow wandering head turn, and now and then a
 * wider smile, all layered on the pose and face shapes the avatar opened in.
 * Returns update(seconds), to call once a frame.
 */
export function idleMotion(root, shapes = TOON.shapes) {
  const bone = (name) => {
    const b = boneNamed(root, name);
    return b ? { b, rest: b.quaternion.clone() } : null;
  };
  const head = bone('neck02'), chest = bone('spine02'), shoulders = ['clavicle.L', 'clavicle.R'].map(bone);
  const faces = [];
  root.traverse((o) => { if (o.morphTargetDictionary) faces.push(o); });
  const shape = (o, name, v) => { const i = o.morphTargetDictionary[name]; if (i !== undefined) o.morphTargetInfluences[i] = v; };
  const base = { eyeWideLeft: 0, eyeWideRight: 0, mouthSmileLeft: 0, ...shapes };
  const turn = new THREE.Quaternion(), euler = new THREE.Euler();
  let nextBlink = 1.5;
  return (t) => {
    // A blink: closed for about a tenth of a second, every 2.5-5.5 s.
    if (t > nextBlink + 0.25) nextBlink = t + 2.5 + 3 * Math.random();
    const blink = Math.max(0, 1 - Math.abs(t - nextBlink - 0.1) / 0.1);
    const smile = base.mouthSmileLeft + 0.3 * Math.max(0, Math.sin(t * 0.35)) ** 6;
    for (const o of faces) {
      shape(o, 'eyeBlinkLeft', blink); shape(o, 'eyeBlinkRight', blink);
      shape(o, 'eyeWideLeft', base.eyeWideLeft * (1 - blink)); shape(o, 'eyeWideRight', base.eyeWideRight * (1 - blink));
      shape(o, 'mouthSmileLeft', smile); shape(o, 'mouthSmileRight', smile);
    }
    const breath = Math.sin(t * 2 * Math.PI / 4.2);                 // one breath every 4.2 s
    if (chest) chest.b.quaternion.copy(chest.rest).multiply(turn.setFromEuler(euler.set(-0.015 * breath, 0, 0)));
    shoulders.forEach((s, i) => s && s.b.quaternion.copy(s.rest).multiply(turn.setFromEuler(euler.set(0, 0, (i ? -1 : 1) * 0.012 * breath))));
    if (head) head.b.quaternion.copy(head.rest).multiply(turn.setFromEuler(euler.set(
      0.03 * Math.sin(t * 0.6) + 0.01 * breath, 0.12 * Math.sin(t * 0.31) + 0.04 * Math.sin(t * 0.83), 0.03 * Math.sin(t * 0.47))));
  };
}

/**
 * Gestures for the cartoon style, played on the avatar's own skeleton. Each
 * is a list of [bone, world axis, angle(t, seconds in)] turns, eased in and
 * out over its length. Turns are about world axes (the avatar faces +z, up is
 * +y, its left hand at +x) so they do not depend on how MakeHuman orients
 * each bone; listed parent first. idleMotion's bones keep breathing under them.
 */
const X = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);
const wave = (hz, t) => Math.sin(2 * Math.PI * hz * t);
export const GESTURES = {
  wave: { seconds: 3.2, turns: [
    ['upperarm01.R', Z, () => -2.15], ['lowerarm01.R', Z, (t) => -0.5 + 0.35 * wave(2.2, t)]] },
  nod: { seconds: 2.0, turns: [['neck01', X, (t) => 0.18 * Math.max(0, wave(1.5, t))]] },
  shrug: { seconds: 2.0, turns: [
    ['clavicle.L', Z, () => 0.22], ['clavicle.R', Z, () => -0.22],
    ['upperarm01.L', Z, () => 0.35], ['upperarm01.R', Z, () => -0.35],
    ['lowerarm01.L', Z, () => 0.9], ['lowerarm01.R', Z, () => -0.9], ['neck01', Z, () => 0.1]] },
  dance: { seconds: 6.0, bounce: 0.12, turns: [
    ['root', Y, (t) => 0.25 * wave(0.8, t)], ['spine03', Z, (t) => 0.1 * wave(1.6, t)],
    ['upperarm01.L', Z, (t) => 1.3 + 0.6 * wave(1.6, t)], ['upperarm01.R', Z, (t) => -1.3 + 0.6 * wave(1.6, t)],
    ['lowerarm01.L', Z, (t) => 0.9], ['lowerarm01.R', Z, (t) => -0.9], ['neck01', Z, (t) => 0.12 * wave(1.6, t + 0.15)]] },
};
const IDLE_BONES = new Set(['neck02', 'spine02', 'clavicle.L', 'clavicle.R']);
export function gestureMotion(root) {
  const hips = boneNamed(root, 'root'), restRoot = hips?.position.clone(), rest = new Map();
  for (const g of Object.values(GESTURES)) for (const [name] of g.turns) {
    const b = boneNamed(root, name);
    if (b && !rest.has(name)) rest.set(name, { b, q: b.quaternion.clone() });
  }
  const world = new THREE.Quaternion(), turn = new THREE.Quaternion(), inverse = new THREE.Quaternion();
  let playing = null, start = 0, now = 0;
  const reset = () => {
    for (const [name, r] of rest) if (!IDLE_BONES.has(name)) r.b.quaternion.copy(r.q);
    if (restRoot) hips.position.copy(restRoot);
  };
  return {
    play(name) { playing = GESTURES[name] || null; start = now; },
    update(seconds) {
      now = seconds;
      reset();
      if (!playing) return;
      const t = seconds - start, p = t / playing.seconds;
      if (p >= 1) { playing = null; return; }
      const ease = Math.min(1, p / 0.15, (1 - p) / 0.15);
      const e = ease * ease * (3 - 2 * ease);
      for (const [name, axis, angle] of playing.turns) {
        const r = rest.get(name);
        if (!r) continue;
        r.b.parent.getWorldQuaternion(world);
        // The turn about a world axis, as the bone's parent sees it.
        turn.setFromAxisAngle(axis, angle(t) * e);
        r.b.quaternion.premultiply(inverse.copy(world).invert().multiply(turn).multiply(world));
      }
      if (playing.bounce && restRoot) hips.position.y = restRoot.y + playing.bounce * Math.abs(wave(0.8, t)) * e;
    },
  };
}

/**
 * Wear one of the pairs of glasses the file carries (its mesh's
 * `glassesOptions`), the others hidden; with none given, none: likeness is
 * judged bare-faced, and the app wears the pair its sidecar json picks.
 */
export function chooseGlasses(root, pair = null) {
  root.traverse((o) => {
    if (o.isMesh && o.userData?.glassesOptions?.includes(o.material?.name)) o.visible = o.material.name === pair;
  });
}

/**
 * Wear one of the haircuts the file carries (its mesh's `hair`, the one the
 * photo read, and `hairOptions`, the rest fitted to the same head), the others
 * hidden; `cut` 'bald' or null wears none. `colour` (sRGB 0-255) recolours it:
 * the file writes hair in skinned_glb.HAIR_GREY and colours it by the
 * material, so the factor is the colour over that grey, in linear terms.
 * With no `cut`, the file's own.
 */
export const HAIR_GREY = 200;
export function chooseHair(root, cut, colour) {
  const linear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  root.traverse((o) => {
    if (!o.isMesh || !o.userData?.hairOptions) return;
    const own = o.userData.hair;
    const name = o.material?.name;
    if (name !== own && !o.userData.hairOptions.includes(name)) return;
    const wearing = (cut === undefined ? own : cut) === name;
    o.visible = wearing;
    if (wearing && colour) {
      o.material.color.setRGB(...colour.map((c) => Math.min(linear(c / 255) / linear(HAIR_GREY / 255), 1)));
    }
  });
}

/**
 * What glTF's materials cannot say about skin and eyes, added at draw time.
 * Calibration knobs; lab renders override them.
 */
export const SHADING = {
  // Diffuse falloff per channel, pow(N.L, k): red light scatters further
  // under the skin, so it fades later towards the shadow side than blue.
  // Zero still at N.L = 0, so it never lights what the shadow map darkens.
  falloff: [0.75, 0.9, 1.0],
  // Light that enters skin comes out again a millimetre or so away, red
  // furthest: how much each channel's diffuse sees of the pore normal map
  // (the rest sees the surface without it). Pores show in blue most.
  pores: [0.3, 0.7, 1.0],
  // Lower lid margin, within `band` eyeball radii of the eye opening: wet,
  // so it catches a thin highlight: this roughness, specular x `gain`.
  // (Region-varying shine is the file's roughness map: T-zone, cheeks, lips.
  // Stronger specular on the T-zone as well was barely visible and made the
  // foreheads that already render too bright brighter still.)
  wet: { roughness: 0.06, gain: 4.0, band: 0.06 },
  // Cornea: a clear coat over the eye, and the iris seen through it from the
  // depth of the anterior chamber (in eyeball radii at the pupil). `coat` is
  // its strength: at 1 the highlight over a light iris was 1.6x the photos'
  // (compose.ROUGHNESS has the numbers). The file's own coat
  // (compose.CORNEA_COAT) wins; these are for files made before it had one.
  cornea: { roughness: 0.03, depth: 0.25, ior: 1.376, coat: 0.3 },
  // Skin is baked from a photo that carries its own light, which de-lighting
  // takes out only in part, so the studio's light on it doubles the shading.
  // Only `lit` of the skin's diffuse takes the lights; the rest is its
  // texture times `baked`, unshaded: delight.RENDER_LIGHT, the light the
  // cheeks get, so the skin keeps its brightness there. Hair, eyes and
  // clothes keep the full studio. Likeness (identity_score.py, five test
  // faces, two builds each) rose 0.471 -> 0.495 at the photo's pose and
  // 0.454 -> 0.474 level, photo-16's two other photos +0.035; the face keeps
  // about 42% of the studio's light and shade. Lower (0.3, 0) scored no
  // better and flattened nose and cheeks; 0.7 gained half as much.
  skin: { lit: 0.5, baked: 2.1 },
  // The key light's shadow on the neck takes this much of the baked half too.
  // Unshaded, the chin's shadow hardly showed and chin and neck were one colour:
  // MediaPipe read the render's chin 3.2% of the eye spacing above the mesh's.
  // Below the jaw only (`jawLine`, from the file): on the face the photo already
  // carries its shading, and the nose's and brows' shadows would come out twice.
  neck: { shadow: 0.85, fade: 0.4 },
  // Pores on the head: a tiled detail normal (poreTile), `repeat` tiles across
  // the head's texture, `strength` strong, on the highlight only (the diffuse
  // keeps its normals, so no pore reads as a mark). The baked photo is smoothed
  // (appearance.smooth_skin) and up close the skin read as wax. A pore is under
  // a pixel at full-head size; the tile's mipmaps average it out there.
  poreDetail: { repeat: 26, strength: 1.2 },
};
// Skin painted, not photographed: more of it lit (the paint has no photo
// shading to double), warmer at the shadow's edge, no pores, and the chin's
// shadow on the neck softer (the realistic 0.85 under the portrait key took
// the neck to 82 37 13 under cheeks at 190 133 103).
export const STYLIZED_SHADING = { ...SHADING, falloff: [0.65, 0.85, 1.0], skin: { lit: 0.7, baked: 1.9 },
                                  neck: { shadow: 0.4, fade: 0.4 }, poreDetail: null };

let poreTexture = null;
/**
 * A tileable pore height field as a normal map: small round dips at jittered
 * points (Worley), with a faint grain between them. Made here, once, so no
 * file ships.
 */
function poreTile(size = 256, cells = 22, seed = 7) {
  if (poreTexture) return poreTexture;
  let t = seed;
  const rand = () => { t = (t + 0x6d2b79f5) | 0; let r = Math.imul(t ^ (t >>> 15), 1 | t); r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r; return ((r ^ (r >>> 14)) >>> 0) / 4294967296; };
  const pts = [];
  for (let j = 0; j < cells; j++) for (let i = 0; i < cells; i++) {
    pts.push([(i + 0.15 + 0.7 * rand()) / cells, (j + 0.15 + 0.7 * rand()) / cells, 0.6 + 0.8 * rand()]);
  }
  const h = new Float32Array(size * size), r = 0.22 / cells;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const u = (x + 0.5) / size, v = (y + 0.5) / size;
    let depth = 0;
    for (const [px, py, k] of pts) {
      let dx = Math.abs(u - px), dy = Math.abs(v - py);
      dx = Math.min(dx, 1 - dx); dy = Math.min(dy, 1 - dy);
      const d2 = (dx * dx + dy * dy) / (r * r * k);
      if (d2 < 9) depth += Math.exp(-d2);
    }
    h[y * size + x] = -depth + 0.15 * (rand() - 0.5);
  }
  const data = new Uint8Array(size * size * 4);
  const at = (x, y) => h[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const nx = at(x - 1, y) - at(x + 1, y), ny = at(x, y - 1) - at(x, y + 1), l = Math.hypot(nx, ny, 1), o = 4 * (y * size + x);
    data[o] = (nx / l * 0.5 + 0.5) * 255; data[o + 1] = (ny / l * 0.5 + 0.5) * 255; data[o + 2] = (1 / l * 0.5 + 0.5) * 255; data[o + 3] = 255;
  }
  poreTexture = new THREE.DataTexture(data, size, size);
  poreTexture.wrapS = poreTexture.wrapT = THREE.RepeatWrapping;
  poreTexture.magFilter = THREE.LinearFilter;
  poreTexture.minFilter = THREE.LinearMipmapLinearFilter;
  poreTexture.generateMipmaps = true;
  poreTexture.needsUpdate = true;
  return poreTexture;
}

const glsl = (v) => `vec3(${v.map((x) => x.toFixed(4)).join(', ')})`;

/** Eyeball centre and radius per eye, in the mesh's bind space (eyes sit either side of x = 0). */
function eyeballs(eyes) {
  const p = eyes.geometry.attributes.position, v = new THREE.Vector3();
  const sides = [-1, 1].map(() => ({ c: new THREE.Vector3(), n: 0, r: 0 }));
  for (let i = 0; i < p.count; i++) { const s = sides[p.getX(i) > 0 ? 1 : 0]; s.c.add(v.fromBufferAttribute(p, i)); s.n++; }
  sides.forEach((s) => s.c.divideScalar(Math.max(s.n, 1)));
  for (let i = 0; i < p.count; i++) { const s = sides[p.getX(i) > 0 ? 1 : 0]; s.r += v.fromBufferAttribute(p, i).distanceTo(s.c) / s.n; }
  return sides.filter((s) => s.n > 0);
}

/**
 * The mesh's open boundary, as [[x, y, z], [x, y, z]] edges: those used by
 * one triangle once vertices are merged by position (the texture splits them
 * at its seams, which would otherwise read as boundary too).
 */
function openEdges(mesh) {
  const p = mesh.geometry.attributes.position, index = mesh.geometry.index;
  const ids = new Map(), at = [], weld = new Int32Array(p.count), uses = new Map();
  for (let i = 0; i < p.count; i++) {
    const k = `${p.getX(i).toFixed(5)},${p.getY(i).toFixed(5)},${p.getZ(i).toFixed(5)}`;
    if (!ids.has(k)) { ids.set(k, at.length); at.push([p.getX(i), p.getY(i), p.getZ(i)]); }
    weld[i] = ids.get(k);
  }
  for (let t = 0; t < index.count; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = weld[index.getX(t + e)], b = weld[index.getX(t + (e + 1) % 3)];
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      uses.set(k, (uses.get(k) || 0) + 1);
    }
  }
  return [...uses].filter(([, n]) => n === 1).map(([k]) => k.split(',').map((v) => at[+v]));
}

const RIM_BINS = 72;

/**
 * The opening each eye shows through: per eye, its centre and radius, and the
 * distance from its axis to the lids' edge, in eyeball radii, in each of
 * RIM_BINS directions round the axis. The opening is a hole in the head mesh,
 * so its edge is the mesh's open boundary there.
 */
function lidRims(head, balls) {
  const open = openEdges(head);
  return balls.map(({ c, r }) => {
    const edges = open.filter((ed) => ed.every(([x, y, z]) => Math.hypot(x - c.x, y - c.y) < 1.5 * r && z > c.z));
    const rim = [];
    for (let i = 0; i < RIM_BINS; i++) {
      const th = ((i + 0.5) / RIM_BINS) * 2 * Math.PI - Math.PI, ux = Math.cos(th), uy = Math.sin(th);
      let best = 0;
      for (const [[ax, ay], [bx, by]] of edges) {
        // c + t u = a + s (b - a)
        const px = ax - c.x, py = ay - c.y, qx = bx - ax, qy = by - ay, den = ux * qy - uy * qx;
        if (Math.abs(den) < 1e-12) continue;
        const t = (px * qy - py * qx) / den, s = (px * uy - py * ux) / den;
        if (t > 0 && s >= 0 && s <= 1 && (!best || t < best)) best = t;
      }
      rim.push(best / r);
    }
    return { eye: new THREE.Vector4(c.x, c.y, c.z, r), rim };
  });
}

/** Per eye vertex, its height above the iris plane (0 off the iris), in mesh units. */
function corneaDepth(eyes, balls, depth) {
  const p = eyes.geometry.attributes.position, h = new Float32Array(p.count), v = new THREE.Vector3();
  const IRIS = Math.cos(THREE.MathUtils.degToRad(28.5));     // facetexture.bake_eyes' iris edge
  for (let i = 0; i < p.count; i++) {
    const { c, r } = balls.length > 1 && p.getX(i) > 0 ? balls[1] : balls[0];
    v.fromBufferAttribute(p, i).sub(c);
    const cos = v.z / v.length();
    h[i] = cos > IRIS ? ((cos - IRIS) / (1 - IRIS)) * depth * r : 0;
  }
  return h;
}

function shadeSkin(m, rims, jawLine, S = SHADING) {
  const wet = (rims.length === 2 && S.wet) || null;
  const neck = (S.neck && jawLine) || null;
  const pores = (S.poreDetail && rims.length === 2 && m.normalMap) ? S.poreDetail : null;
  if (S.skin && m.map) {
    m.color.multiplyScalar(S.skin.lit);
    m.emissiveMap = m.map;
    m.emissive.setScalar(S.skin.baked * (1 - S.skin.lit));
  }
  m.onBeforeCompile = (shader) => {
    if (pores) shader.uniforms.poreTile = { value: poreTile() };
    if (wet) {
      shader.uniforms.lidEye = { value: rims.map((x) => x.eye) };
      shader.uniforms.lidRim = { value: new Float32Array(rims.flatMap((x) => x.rim)) };
    }
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vBindPos;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvBindPos = position;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
vec3 skinN[ 3 ];   // the normal each channel's diffuse sees
varying vec3 vBindPos;
${pores ? 'uniform sampler2D poreTile;' : ''}
${wet ? `uniform vec4 lidEye[2];
uniform float lidRim[${2 * RIM_BINS}];
// 1 on the lower lid's inner edge (within ${wet.band} eyeball radii of the
// opening, or inside it), fading out towards the corners; 0 elsewhere.
float lidWet() {
  float w = 0.0;
  for ( int e = 0; e < 2; e ++ ) {
    vec3 d = vBindPos - lidEye[ e ].xyz;
    float r = length( d.xy ) / lidEye[ e ].w;
    float a = ( atan( d.y, d.x ) + PI ) / PI2 * ${RIM_BINS}.0 - 0.5;
    int i = int( floor( a ) ) + ${RIM_BINS};
    float rim = mix( lidRim[ e * ${RIM_BINS} + i % ${RIM_BINS} ], lidRim[ e * ${RIM_BINS} + ( i + 1 ) % ${RIM_BINS} ], fract( a ) );
    float onRim = 1.0 - smoothstep( 0.0, ${wet.band.toFixed(4)}, r - rim );
    float below = clamp( - d.y / ( 0.25 * lidEye[ e ].w ), 0.0, 1.0 );
    w = max( w, onRim * below * step( 0.0, d.z ) * step( r, 1.2 ) * step( 0.01, rim ) );
  }
  return w;
}` : 'float lidWet() { return 0.0; }'}`)
      .replace('#include <lights_physical_pars_fragment>', `#include <lights_physical_pars_fragment>
void RE_Direct_Skin( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
  vec3 irradiance = saturate( dot( geometryNormal, directLight.direction ) ) * directLight.color;
  #ifdef USE_SHEEN
    sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
  #endif
  reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
  vec3 nl = saturate( vec3( dot( skinN[ 0 ], directLight.direction ), dot( skinN[ 1 ], directLight.direction ), dot( skinN[ 2 ], directLight.direction ) ) );
  reflectedLight.directDiffuse += pow( nl, ${glsl(S.falloff)} ) * directLight.color * BRDF_Lambert( material.diffuseColor );
}
#undef RE_Direct
#define RE_Direct RE_Direct_Skin`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
float skinWet = lidWet();
roughnessFactor = mix( roughnessFactor, ${(wet ? wet.roughness : 0).toFixed(4)}, skinWet );`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
${S.pores.map((w, c) => `skinN[ ${c} ] = normalize( mix( nonPerturbedNormal, normal, ${w.toFixed(4)} ) );`).join('\n')}
${pores ? `#ifdef USE_NORMALMAP_TANGENTSPACE
  vec2 poreUv = vNormalMapUv * ${pores.repeat.toFixed(4)};
  vec3 poreN = texture2D( poreTile, poreUv ).xyz * 2.0 - 1.0;
  normal = normalize( tbn * normalize( vec3( mapN.xy + poreN.xy * ${pores.strength.toFixed(4)}, mapN.z ) ) );
#endif` : ''}`)
      .replace('#include <lights_physical_fragment>', `#include <lights_physical_fragment>
material.specularColor *= mix( 1.0, ${(wet ? wet.gain : 1).toFixed(4)}, skinWet );`)
      .replace('#include <aomap_fragment>', neck ? `#include <aomap_fragment>
#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
  float belowJaw = 1.0 - smoothstep( ${neck[0].toFixed(4)}, ${(neck[0] + S.neck.fade * neck[1]).toFixed(4)}, vBindPos.y );
  float keyShadow = receiveShadow ? getShadow( directionalShadowMap[ 0 ], directionalLightShadows[ 0 ].shadowMapSize, directionalLightShadows[ 0 ].shadowBias, directionalLightShadows[ 0 ].shadowRadius, vDirectionalShadowCoord[ 0 ] ) : 1.0;
  totalEmissiveRadiance *= 1.0 - ${S.neck.shadow.toFixed(4)} * belowJaw * ( 1.0 - keyShadow );
#endif` : '#include <aomap_fragment>')
      .replace('#include <lights_fragment_maps>', THREE.ShaderChunk.lights_fragment_maps.replace(
        'iblIrradiance += getIBLIrradiance( geometryNormal );',
        'iblIrradiance += vec3( getIBLIrradiance( skinN[ 0 ] ).r, getIBLIrradiance( skinN[ 1 ] ).g, getIBLIrradiance( skinN[ 2 ] ).b );'));
  };
  const key = 'skin' + JSON.stringify(S) + Boolean(wet) + (neck ? neck.join() : '') + Boolean(pores);
  m.customProgramCacheKey = () => key;
}

function shadeEyes(m, coat = {}) {
  const S = SHADING.cornea;
  m.clearcoat = coat.clearcoat || S.coat;
  m.clearcoatRoughness = coat.clearcoat ? coat.clearcoatRoughness : S.roughness;
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float corneaDepth;\nvarying float vCornea;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvCornea = corneaDepth;');
    // The iris lies behind the cornea: follow the refracted view ray down to
    // it and read the texture where it lands, moved along the surface.
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vCornea;')
      .replace('#include <map_fragment>', `#ifdef USE_MAP
  vec2 eyeUv = vMapUv;
  vec3 en = normalize( vNormal ), ev = normalize( vViewPosition );
  vec3 q0 = dFdx( - vViewPosition ), q1 = dFdy( - vViewPosition );
  vec2 s0 = dFdx( vMapUv ), s1 = dFdy( vMapUv );
  vec3 q1p = cross( q1, en ), q0p = cross( en, q0 );
  float det = dot( q0, q1p );
  if ( vCornea > 0.0 && abs( det ) > 1e-12 ) {
    vec3 ray = refract( - ev, en, ${(1 / S.ior).toFixed(4)} );
    vec3 travel = ( ray - dot( ray, en ) * en ) * vCornea / max( - dot( ray, en ), 0.3 );
    // The chamber thins to nothing at the iris's edge; travelling further
    // than it is deep would read the white from beyond the edge.
    travel *= min( 1.0, 0.8 * vCornea / max( length( travel ), 1e-6 ) );
    eyeUv += vec2( dot( travel, q1p * s0.x + q0p * s1.x ), dot( travel, q1p * s0.y + q0p * s1.y ) ) / det;
  }
  diffuseColor *= texture2D( map, eyeUv );
#endif`);
  };
  const key = 'eyes' + JSON.stringify(S);
  m.customProgramCacheKey = () => key;
}

function shadeSkinAndEyes(root, shading = SHADING) {
  const meshes = [];
  root.traverse((o) => { if (o.isMesh && !Array.isArray(o.material)) meshes.push(o); });
  const eyes = meshes.find((o) => o.material.name === 'eyes');
  const balls = eyes ? eyeballs(eyes) : [];
  const skins = meshes.filter((o) => ['head', 'body'].includes(o.material.name) && o.material.isMeshPhysicalMaterial);
  for (const o of meshes) {
    const name = o.material.name;
    if (skins.includes(o)) {
      shadeSkin(o.material, name === 'head' ? lidRims(o, balls) : [], o.userData?.jawLine, shading);
    } else if (o === eyes && SHADING.cornea) {
      // A clear coat needs the physical material; the file's eyes are plain PBR.
      const m = new THREE.MeshPhysicalMaterial();
      THREE.MeshStandardMaterial.prototype.copy.call(m, o.material);
      m.defines = { STANDARD: '', PHYSICAL: '' };
      const coat = { clearcoat: o.material.clearcoat, clearcoatRoughness: o.material.clearcoatRoughness };
      o.material.dispose();
      o.material = m;
      o.geometry.setAttribute('corneaDepth', new THREE.BufferAttribute(corneaDepth(o, balls, SHADING.cornea.depth), 1));
      shadeEyes(m, coat);
    }
  }
}
