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
  const ao = new GTAOPass(scene, camera, size.x, size.y);
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

export function setupRealism(renderer, scene, { exposure = 1.05 } = {}) {
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
  scene.environmentRotation.set(THREE.MathUtils.degToRad(30), 0, 0);

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
}
