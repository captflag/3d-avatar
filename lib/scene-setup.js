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
 */
export function withOcclusion(renderer, scene, camera, { radius = 0.6, intensity = 0.9 } = {}) {
  const size = renderer.getSize(new THREE.Vector2());
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const ao = new GTAOPass(scene, camera, size.x, size.y);
  ao.updateGtaoMaterial({ radius, distanceExponent: 1.5, thickness: 1.0, scale: 1.0, samples: 16 });
  ao.blendIntensity = intensity;
  composer.addPass(ao);
  composer.addPass(new OutputPass());       // tone mapping and sRGB, as a plain render would
  return {
    render: () => composer.render(),
    setSize: (w, h) => composer.setSize(w, h),
    dispose: () => composer.dispose(),
  };
}

export function setupRealism(renderer, scene, { exposure = 1.05 } = {}) {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = exposure;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  // Key: front and to one side, the light that models the face and casts the
  // shadow under the chin and between the arms.
  const key = new THREE.DirectionalLight(0xfff2e6, 1.7);
  key.position.set(1.6, 2.4, 2.6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 4;
  scene.add(key);
  scene.add(key.target);

  // Fill: cool and weak, so the shadow side reads as shadow rather than black.
  const fill = new THREE.DirectionalLight(0xd6e2ff, 0.4);
  fill.position.set(-2.0, 0.6, 1.2);
  scene.add(fill);

  // Rim: behind and above. This is the bright outline on shoulders, hair and
  // jaw in studio portraits, and the main thing that lifts a figure off a dark
  // background instead of letting it sink into it.
  const rim = new THREE.DirectionalLight(0xffffff, 1.4);
  rim.position.set(-0.6, 2.2, -3.0);
  scene.add(rim);

  return { pmrem, key };
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
  key.position.set(reach * 0.35, reach * 0.55, reach * 0.6);
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
    }
  });
}
