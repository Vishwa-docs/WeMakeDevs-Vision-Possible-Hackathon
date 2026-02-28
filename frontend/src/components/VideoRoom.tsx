/**
 * WorldLens — Video Room component
 * Connects to a Stream Video call and renders local + remote video.
 *
 * Simplified: single useEffect handles the entire lifecycle (token → client →
 * join → media) so there are no competing cleanup functions that can
 * accidentally disconnect the user mid-session.
 */
import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
  ParticipantView,
  useCallStateHooks,
  StreamTheme,
  CallingState,
} from "@stream-io/video-react-sdk";
import type { StreamVideoParticipant } from "@stream-io/video-react-sdk";
import "@stream-io/video-react-sdk/dist/css/styles.css";
import React, { Suspense, useEffect, useState, useCallback, useRef } from "react";
import { STREAM_API_KEY, getStreamConfig } from "../utils/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface VideoRoomProps {
  callId: string;
  callType?: string;
  userId?: string;
  userName?: string;
  onLeave?: () => void;
}

// ---------------------------------------------------------------------------
// ErrorBoundary — prevents Stream SDK crashes from white-screening the app
// ---------------------------------------------------------------------------
interface EBProps {
  children: React.ReactNode;
  onReset?: () => void;
}
interface EBState {
  error: Error | null;
}

class VideoErrorBoundary extends React.Component<EBProps, EBState> {
  constructor(props: EBProps) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[WorldLens][VideoErrorBoundary]", error, info.componentStack);
  }
  render() {
    if (this.state.error) {
      return (
        <div className="video-room error">
          <p>⚠️ Video stream crashed: {this.state.error.message}</p>
          <button
            className="btn btn-primary"
            onClick={() => {
              this.setState({ error: null });
              this.props.onReset?.();
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ---------------------------------------------------------------------------
// RemoteAudio — plays audio for a single remote participant via <audio> tag.
// This replaces ParticipantsAudio which crashes when participants is undefined.
// ---------------------------------------------------------------------------
function RemoteAudio({ participant }: { participant: StreamVideoParticipant }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioStream = participant.audioStream;

  useEffect(() => {
    const el = audioRef.current;
    if (!el || !audioStream) return;
    el.srcObject = audioStream;
    el.play().catch(() => {});
    return () => {
      el.srcObject = null;
    };
  }, [audioStream]);

  if (!audioStream) return null;
  return <audio ref={audioRef} autoPlay playsInline />;
}

// ---------------------------------------------------------------------------
// Inner component (must be inside StreamCall provider)
// All hooks called unconditionally at top level (React rules of hooks).
// Rendering is gated on CallingState.JOINED so SDK state is populated.
// ---------------------------------------------------------------------------
function CallUI({ onLeave }: { onLeave?: () => void }) {
  // Call ALL hooks unconditionally at the top — React rules of hooks.
  // useCallStateHooks() returns the hooks module; we then call each hook.
  let callingState: CallingState | undefined;
  let participants: StreamVideoParticipant[] | undefined;
  let localParticipant: StreamVideoParticipant | undefined;

  try {
    const hooks = useCallStateHooks();
    callingState = hooks.useCallCallingState();
    participants = hooks.useParticipants();
    localParticipant = hooks.useLocalParticipant();
  } catch (err) {
    console.error("[WorldLens][CallUI] Hook error:", err);
    return (
      <div className="call-ui">
        <div className="video-grid">
          <div
            className="video-tile remote"
            style={{ display: "grid", placeItems: "center", color: "#ff6b6b" }}
          >
            <span>⚠️ Stream SDK error: {String(err)}</span>
          </div>
        </div>
        <div className="call-controls">
          <button className="btn btn-danger" onClick={onLeave}>
            Leave Call
          </button>
        </div>
      </div>
    );
  }

  // Gate: only render participant-dependent UI once the call is fully joined
  // and the SDK has initialized the participants array.
  if (
    callingState !== CallingState.JOINED ||
    !participants ||
    !Array.isArray(participants)
  ) {
    return (
      <div className="call-ui">
        <div className="video-grid">
          <div
            className="video-tile remote"
            style={{ display: "grid", placeItems: "center", color: "#999" }}
          >
            <span>
              {callingState === CallingState.JOINING
                ? "Joining call…"
                : callingState === CallingState.RECONNECTING
                  ? "Reconnecting…"
                  : "Initialising call…"}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Separate remote participants (agent) from local
  const remoteParticipants = participants.filter(
    (p) => p.sessionId !== localParticipant?.sessionId
  );
  const hasAnyVideoParticipant =
    Boolean(localParticipant) || remoteParticipants.length > 0;

  return (
    <div className="call-ui">
      {/* Main video area — side by side: AI (large) + You (small) */}
      <div className="video-grid">
        {!hasAnyVideoParticipant && (
          <div
            className="video-tile remote"
            style={{ display: "grid", placeItems: "center", color: "#999" }}
          >
            <span>Joining call… waiting for camera/participants.</span>
          </div>
        )}

        {/* Remote participants (agent AI video) — main large view */}
        {remoteParticipants.map((p) => (
          <div key={p.sessionId} className="video-tile remote">
            <ParticipantView participant={p} trackType="videoTrack" />
            <span className="video-label">
              {p.name || p.userId || "WorldLens AI"}
            </span>
          </div>
        ))}

        {/* Local camera (user) — small picture-in-picture */}
        {localParticipant && (
          <div className="video-tile local">
            <ParticipantView
              participant={localParticipant}
              trackType="videoTrack"
            />
            <span className="video-label">You</span>
          </div>
        )}
      </div>

      {/* Play remote participants' audio (agent voice) */}
      {remoteParticipants.map((p) => (
        <RemoteAudio key={`audio-${p.sessionId}`} participant={p} />
      ))}

      {/* Controls */}
      <div className="call-controls">
        <button className="btn btn-danger btn-end-session" onClick={onLeave}>
          End Session
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main VideoRoom component
// ---------------------------------------------------------------------------
export function VideoRoom({
  callId,
  callType = "default",
  userId = "user-web",
  userName = "Web User",
  onLeave,
}: VideoRoomProps) {
  console.log("[WorldLens][VideoRoom] render", { callId, callType, userId });

  const [client, setClient] = useState<StreamVideoClient | null>(null);
  const [call, setCall] = useState<ReturnType<
    StreamVideoClient["call"]
  > | null>(null);
  const [joined, setJoined] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Keep refs so the unmount cleanup always sees the latest objects
  const clientRef = useRef<StreamVideoClient | null>(null);
  const callRef = useRef<ReturnType<StreamVideoClient["call"]> | null>(null);

  // ------------------------------------------------------------------
  // Single consolidated effect: token → client → join → enable media
  // Only ONE cleanup function = no accidental disconnects.
  // ------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        // 1. Resolve API key (env first, then backend fallback)
        let key = STREAM_API_KEY;
        if (!key) {
          console.log("[WorldLens][VideoRoom] apiKey missing, fetching from backend");
          const cfg = await getStreamConfig();
          key = cfg.api_key || "";
        }
        if (!key) {
          throw new Error("No Stream API key configured (VITE_STREAM_API_KEY)");
        }
        if (cancelled) return;

        // 2. Fetch user token from backend
        const backendUrl =
          import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
        console.log("[WorldLens][VideoRoom] fetching token", { userId, callId });
        const res = await fetch(`${backendUrl}/token?user_id=${userId}`);
        if (!res.ok)
          throw new Error(`Token endpoint failed: ${res.status} ${res.statusText}`);
        const data = await res.json();
        if (!data?.token) throw new Error("Token endpoint returned empty token");
        if (cancelled) return;

        // 3. Create Stream Video client
        // Use tokenProvider when possible for resilience (SDK can refresh)
        const newClient = new StreamVideoClient({
          apiKey: key,
          user: { id: userId, name: userName },
          token: data.token,
          tokenProvider: async () => {
            const r = await fetch(`${backendUrl}/token?user_id=${userId}`);
            const d = await r.json();
            return d.token;
          },
        });
        clientRef.current = newClient;
        console.log("[WorldLens][VideoRoom] StreamVideoClient created");
        if (cancelled) {
          newClient.disconnectUser();
          return;
        }

        // 4. Create call object and join
        const streamCall = newClient.call(callType, callId);
        callRef.current = streamCall;
        console.log("[WorldLens][VideoRoom] joining call", { callType, callId });

        await streamCall.join({ create: true });
        if (cancelled) {
          await streamCall.leave().catch(() => {});
          newClient.disconnectUser();
          return;
        }
        console.log("[WorldLens][VideoRoom] join success");

        // 5. Enable camera & microphone (best-effort)
        // Use 480p for lower latency — reduces bandwidth & backend processing
        try {
          await streamCall.camera.enable();
          // Prefer 480p @ 15fps for lower latency over quality
          try {
            const videoTrack = streamCall.camera.state.mediaStream?.getVideoTracks()[0];
            if (videoTrack) {
              await videoTrack.applyConstraints({
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 15, max: 20 },
              });
              console.log("[WorldLens][VideoRoom] camera constrained to 480p@15fps");
            }
          } catch (constraintErr) {
            console.warn("[WorldLens][VideoRoom] camera constraint failed", constraintErr);
          }
          console.log("[WorldLens][VideoRoom] camera enabled");
        } catch (e) {
          console.warn("[WorldLens][VideoRoom] camera enable failed", e);
        }
        try {
          await streamCall.microphone.enable();
          console.log("[WorldLens][VideoRoom] microphone enabled");
        } catch (e) {
          console.warn("[WorldLens][VideoRoom] microphone enable failed", e);
        }

        if (cancelled) {
          await streamCall.leave().catch(() => {});
          newClient.disconnectUser();
          return;
        }

        // 6. All ready — update state in one batch
        setClient(newClient);
        setCall(streamCall);
        setJoined(true);
        console.log("[WorldLens][VideoRoom] fully initialised");
      } catch (err) {
        if (!cancelled) {
          console.error("[WorldLens][VideoRoom] init failed", err);
          setError(
            err instanceof Error ? err.message : "Failed to initialise video"
          );
        }
      }
    }

    init();

    // Cleanup: runs ONLY when callId changes (shouldn't) or on unmount
    return () => {
      cancelled = true;
      console.log("[WorldLens][VideoRoom] cleanup — leaving call & disconnecting");
      callRef.current?.leave().catch(() => {});
      callRef.current = null;
      clientRef.current?.disconnectUser();
      clientRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [callId]);

  // ------------------------------------------------------------------
  // Manual leave handler (via Leave Call button)
  // ------------------------------------------------------------------
  const handleLeave = useCallback(async () => {
    console.log("[WorldLens][VideoRoom] manual leave requested");
    if (callRef.current) {
      await callRef.current.leave().catch(() => {});
      callRef.current = null;
    }
    if (clientRef.current) {
      clientRef.current.disconnectUser();
      clientRef.current = null;
    }
    setJoined(false);
    setClient(null);
    setCall(null);
    onLeave?.();
  }, [onLeave]);

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  if (error) {
    return (
      <div className="video-room error">
        <p>⚠️ {error}</p>
        <p>Check your .env configuration and backend status.</p>
      </div>
    );
  }

  if (!client || !call || !joined) {
    return (
      <div className="video-room loading">
        <div className="spinner" />
        <p>Connecting to WorldLens…</p>
      </div>
    );
  }

  return (
    <VideoErrorBoundary onReset={onLeave}>
      <Suspense
        fallback={
          <div className="video-room loading">
            <div className="spinner" />
            <p>Loading video components…</p>
          </div>
        }
      >
        <StreamVideo client={client}>
          <StreamTheme>
            <StreamCall call={call}>
              <CallUI onLeave={handleLeave} />
            </StreamCall>
          </StreamTheme>
        </StreamVideo>
      </Suspense>
    </VideoErrorBoundary>
  );
}
