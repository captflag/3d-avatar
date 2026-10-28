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
  for (const side of [1, -1]) {
    const wrap = new THREE.DirectionalLight(0xffffff, 2.0);
    wrap.position.set(side * 3.0 * Math.sin(THREE.MathUtils.degToRad(100)), 0, 3.0 * Math.cos(THREE.MathUtils.degToRad(100)));
    scene.add(wrap);
  }

  return { pmrem, key, fill, rim };
}

/**
 * Fit the key light's shadow camera round the model and put a ground under it.
 *
 * The ground uses ShadowMaterial, which draws only the shadow and is otherwise
 * transparent, so a PNG export still has a clean alpha background but the feet
 * now touch something.
 */
export function groundAndShadow(scene, model, key) {
  const box = new THREE.Box3().setFromObject(model);
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
export function applyMaterials(root) {
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
  shadeSkinAndEyes(root);
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
};

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

function shadeSkin(m, rims) {
  const S = SHADING, wet = (rims.length === 2 && S.wet) || null;
  if (S.skin && m.map) {
    m.color.multiplyScalar(S.skin.lit);
    m.emissiveMap = m.map;
    m.emissive.setScalar(S.skin.baked * (1 - S.skin.lit));
  }
  m.onBeforeCompile = (shader) => {
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
${S.pores.map((w, c) => `skinN[ ${c} ] = normalize( mix( nonPerturbedNormal, normal, ${w.toFixed(4)} ) );`).join('\n')}`)
      .replace('#include <lights_physical_fragment>', `#include <lights_physical_fragment>
material.specularColor *= mix( 1.0, ${(wet ? wet.gain : 1).toFixed(4)}, skinWet );`)
      .replace('#include <lights_fragment_maps>', THREE.ShaderChunk.lights_fragment_maps.replace(
        'iblIrradiance += getIBLIrradiance( geometryNormal );',
        'iblIrradiance += vec3( getIBLIrradiance( skinN[ 0 ] ).r, getIBLIrradiance( skinN[ 1 ] ).g, getIBLIrradiance( skinN[ 2 ] ).b );'));
  };
  const key = 'skin' + JSON.stringify(S) + Boolean(wet);
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

function shadeSkinAndEyes(root) {
  const meshes = [];
  root.traverse((o) => { if (o.isMesh && !Array.isArray(o.material)) meshes.push(o); });
  const eyes = meshes.find((o) => o.material.name === 'eyes');
  const balls = eyes ? eyeballs(eyes) : [];
  const skins = meshes.filter((o) => ['head', 'body'].includes(o.material.name) && o.material.isMeshPhysicalMaterial);
  for (const o of meshes) {
    const name = o.material.name;
    if (skins.includes(o)) {
      shadeSkin(o.material, name === 'head' ? lidRims(o, balls) : []);
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
