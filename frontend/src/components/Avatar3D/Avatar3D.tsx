/**
 * Avatar3D — 3D Avatar with Lip-Sync (SignBridge mode)
 * ====================================================
 * Three.js canvas with lip-sync driven by morph targets.
 *
 * NOTE: Ready Player Me was discontinued on January 31, 2026.
 * The component now defaults to a built-in geometric fallback avatar.
 * If you have your own .glb model with viseme morph targets, you can
 * still supply it via the `avatarUrl` prop or VITE_AVATAR_URL env var.
 *
 * Lip-sync approach:
 *   - The avatar's jaw morph target ("viseme_aa" / "jawOpen") is driven
 *     by a simple oscillator when the agent is speaking.
 *   - When the agent goes silent, the jaw smoothly closes.
 *   - This gives a convincing lip-sync effect without needing raw audio
 *     frequency data from the WebRTC stream.
 *
 * The `isSpeaking` prop should be set to true whenever the agent is
 * producing speech (detected via transcript activity or audio stream).
 */
import React, { useRef, useEffect, Suspense, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment } from "@react-three/drei";
import * as THREE from "three";

// ---------------------------------------------------------------------------
// Custom avatar URL (optional)
// Ready Player Me was discontinued on Jan 31, 2026 — their model URLs
// no longer resolve. If you have your own .glb with viseme morph targets,
// set VITE_AVATAR_URL in frontend/.env. Otherwise the built-in geometric
// fallback avatar is used automatically.
// ---------------------------------------------------------------------------
const CUSTOM_AVATAR_URL: string | undefined =
  import.meta.env.VITE_AVATAR_URL || undefined;

// ---------------------------------------------------------------------------
// Viseme morph target names (Ready Player Me standard)
// ---------------------------------------------------------------------------
const VISEME_NAMES = [
  "viseme_aa",
  "viseme_E",
  "viseme_I",
  "viseme_O",
  "viseme_U",
  "viseme_PP",
  "viseme_FF",
];

// ---------------------------------------------------------------------------
// AvatarModel — loads and animates the GLB
// ---------------------------------------------------------------------------
interface AvatarModelProps {
  url: string;
  isSpeaking: boolean;
}

function AvatarModel({ url, isSpeaking }: AvatarModelProps) {
  const { scene } = useGLTF(url);
  const meshesRef = useRef<THREE.SkinnedMesh[]>([]);
  const timeRef = useRef(0);
  const jawOpenRef = useRef(0);

  // Find all skinned meshes with morph targets
  useEffect(() => {
    const meshes: THREE.SkinnedMesh[] = [];
    scene.traverse((child) => {
      if (
        child instanceof THREE.SkinnedMesh &&
        child.morphTargetDictionary &&
        child.morphTargetInfluences
      ) {
        meshes.push(child);
      }
    });
    meshesRef.current = meshes;

    if (meshes.length === 0) {
      console.warn(
        "[Avatar3D] No skinned meshes with morph targets found in model"
      );
    } else {
      console.log(
        "[Avatar3D] Found morph targets:",
        meshes.map((m) => Object.keys(m.morphTargetDictionary || {}))
      );
    }
  }, [scene]);

  // Animate morph targets every frame
  useFrame((_, delta) => {
    timeRef.current += delta;

    // Target jaw opening — oscillates while speaking
    let targetJaw = 0;
    if (isSpeaking) {
      // Multi-frequency oscillation for more natural mouth movement
      const t = timeRef.current;
      const primary = Math.sin(t * 8) * 0.4 + 0.4; // Main jaw movement
      const secondary = Math.sin(t * 13) * 0.15; // High-freq variation
      const tertiary = Math.sin(t * 3) * 0.1; // Slow modulation
      targetJaw = Math.max(0, Math.min(1, primary + secondary + tertiary));
    }

    // Smooth interpolation (quick open, slower close)
    const speed = isSpeaking ? 12 : 6;
    jawOpenRef.current += (targetJaw - jawOpenRef.current) * speed * delta;

    // Apply to all morphable meshes
    for (const mesh of meshesRef.current) {
      const dict = mesh.morphTargetDictionary;
      const influences = mesh.morphTargetInfluences;
      if (!dict || !influences) continue;

      // Primary jaw morph targets
      const jawTargets = ["jawOpen", "viseme_aa"];
      for (const name of jawTargets) {
        if (name in dict) {
          influences[dict[name]] = jawOpenRef.current;
        }
      }

      // Secondary visemes — smaller random values while speaking
      if (isSpeaking) {
        const t = timeRef.current;
        for (let i = 0; i < VISEME_NAMES.length; i++) {
          const vname = VISEME_NAMES[i];
          if (vname in dict && vname !== "viseme_aa") {
            // Each viseme gets a unique phase offset
            const val =
              Math.max(0, Math.sin(t * (6 + i * 2.5) + i * 1.3)) * 0.3;
            influences[dict[vname]] = val;
          }
        }
      } else {
        // Reset all visemes when not speaking
        for (const vname of VISEME_NAMES) {
          if (vname in dict && vname !== "viseme_aa") {
            const idx = dict[vname];
            influences[idx] *= 0.9; // Gentle fade out
          }
        }
      }
    }
  });

  return (
    <primitive
      object={scene}
      position={[0, -1.5, 0]}
      scale={1.8}
      rotation={[0, 0, 0]}
    />
  );
}

// ---------------------------------------------------------------------------
// Loading fallback
// ---------------------------------------------------------------------------
function AvatarLoader() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial color="#4a90d9" wireframe />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Fallback avatar when GLB fails to load (animated head + body silhouette)
// ---------------------------------------------------------------------------
function FallbackAvatar({ isSpeaking }: { isSpeaking: boolean }) {
  const jawRef = useRef(0);
  const headRef = useRef<THREE.Mesh>(null);
  const timeRef = useRef(0);

  useFrame((_, delta) => {
    timeRef.current += delta;
    const t = timeRef.current;

    // Gentle idle head sway
    if (headRef.current) {
      headRef.current.rotation.y = Math.sin(t * 0.5) * 0.05;
      headRef.current.rotation.z = Math.sin(t * 0.3) * 0.02;
    }

    // Jaw animation when speaking
    if (isSpeaking) {
      jawRef.current = Math.max(0, Math.sin(t * 8) * 0.12 + 0.06);
    } else {
      jawRef.current *= 0.9;
    }
  });

  return (
    <group position={[0, -0.3, 0]}>
      {/* Body / torso */}
      <mesh position={[0, -0.9, 0]}>
        <capsuleGeometry args={[0.4, 0.6, 8, 16]} />
        <meshStandardMaterial color="#3a3a5c" roughness={0.7} />
      </mesh>
      {/* Neck */}
      <mesh position={[0, -0.25, 0]}>
        <cylinderGeometry args={[0.12, 0.15, 0.2, 12]} />
        <meshStandardMaterial color="#6a6a8a" roughness={0.6} />
      </mesh>
      {/* Head */}
      <mesh ref={headRef} position={[0, 0.15, 0]}>
        <sphereGeometry args={[0.32, 24, 24]} />
        <meshStandardMaterial color="#7a7aa0" roughness={0.5} />
      </mesh>
      {/* Eyes */}
      <mesh position={[-0.1, 0.2, 0.28]}>
        <sphereGeometry args={[0.04, 12, 12]} />
        <meshStandardMaterial color="#e0e0ff" emissive="#6c63ff" emissiveIntensity={0.5} />
      </mesh>
      <mesh position={[0.1, 0.2, 0.28]}>
        <sphereGeometry args={[0.04, 12, 12]} />
        <meshStandardMaterial color="#e0e0ff" emissive="#6c63ff" emissiveIntensity={0.5} />
      </mesh>
    </group>
  );
}

// ---------------------------------------------------------------------------
// Error boundary for 3D avatar load failures
// ---------------------------------------------------------------------------
interface AvatarErrorBoundaryProps {
  children: React.ReactNode;
  isSpeaking: boolean;
}
interface AvatarErrorBoundaryState {
  hasError: boolean;
}

class AvatarErrorBoundary extends React.Component<
  AvatarErrorBoundaryProps,
  AvatarErrorBoundaryState
> {
  constructor(props: AvatarErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(error: Error) {
    console.warn("[Avatar3D] GLB load failed, using fallback:", error.message);
  }
  render() {
    if (this.state.hasError) {
      return <FallbackAvatar isSpeaking={this.props.isSpeaking} />;
    }
    return this.props.children;
  }
}

// ---------------------------------------------------------------------------
// SafeAvatarModel — catches GLB load errors gracefully
// ---------------------------------------------------------------------------
function SafeAvatarModel({ url, isSpeaking }: AvatarModelProps) {
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    // Pre-check the URL with a HEAD request
    fetch(url, { method: "HEAD" })
      .then((res) => {
        if (!res.ok) setLoadFailed(true);
      })
      .catch(() => setLoadFailed(true));
  }, [url]);

  if (loadFailed) {
    return <FallbackAvatar isSpeaking={isSpeaking} />;
  }

  return (
    <AvatarErrorBoundary isSpeaking={isSpeaking}>
      <AvatarModel url={url} isSpeaking={isSpeaking} />
    </AvatarErrorBoundary>
  );
}

// ---------------------------------------------------------------------------
// Main Avatar3D Component
// ---------------------------------------------------------------------------
export interface Avatar3DProps {
  /** Whether the agent is currently speaking */
  isSpeaking: boolean;
  /** URL to a custom .glb model with viseme morph targets */
  avatarUrl?: string;
  /** CSS style override for container */
  style?: React.CSSProperties;
  /** CSS class for container */
  className?: string;
}

export function Avatar3D({
  isSpeaking,
  avatarUrl,
  style,
  className = "",
}: Avatar3DProps) {
  const url = avatarUrl || CUSTOM_AVATAR_URL;

  // Preload the model (only if a URL is provided)
  useEffect(() => {
    if (url) useGLTF.preload(url);
  }, [url]);

  return (
    <div
      className={`avatar-3d-container ${className}`}
      style={{
        width: "100%",
        height: "100%",
        minHeight: 200,
        borderRadius: 12,
        overflow: "hidden",
        background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
        ...style,
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 2.2], fov: 35 }}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
        }}
        dpr={[1, 2]}
      >
        {/* Lighting */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[2, 3, 4]} intensity={1.2} castShadow />
        <directionalLight
          position={[-2, 1, -1]}
          intensity={0.3}
          color="#a0c4ff"
        />
        <pointLight position={[0, 2, 3]} intensity={0.4} color="#e0e0ff" />

        {/* Environment for realistic reflections */}
        <Environment preset="city" />

        {/* Avatar — use custom GLB if provided, otherwise geometric fallback */}
        <Suspense fallback={<AvatarLoader />}>
          {url ? (
            <SafeAvatarModel url={url} isSpeaking={isSpeaking} />
          ) : (
            <FallbackAvatar isSpeaking={isSpeaking} />
          )}
        </Suspense>

        {/* Camera controls (limited rotation for a nice framing) */}
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 2}
          minAzimuthAngle={-Math.PI / 6}
          maxAzimuthAngle={Math.PI / 6}
        />
      </Canvas>

      {/* Speaking indicator */}
      {isSpeaking && (
        <div
          style={{
            position: "absolute",
            bottom: 8,
            left: "50%",
            transform: "translateX(-50%)",
            display: "flex",
            gap: 3,
            alignItems: "flex-end",
            height: 16,
          }}
        >
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              style={{
                width: 3,
                borderRadius: 2,
                background: "#4a90d9",
                animation: `avatarSpeakBar 0.6s ease-in-out ${i * 0.1}s infinite alternate`,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default Avatar3D;
