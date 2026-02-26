/**
 * WorldLens — Video Room component
 * Connects to a Stream Video call and renders local + remote video.
 */
import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
  ParticipantView,
  ParticipantsAudio,
  useCallStateHooks,
  StreamTheme,
} from "@stream-io/video-react-sdk";
import "@stream-io/video-react-sdk/dist/css/styles.css";
import { useEffect, useMemo, useState, useCallback } from "react";
import { STREAM_API_KEY } from "../utils/api";

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
// Inner component (must be inside StreamCall provider)
// ---------------------------------------------------------------------------
function CallUI({ onLeave }: { onLeave?: () => void }) {
  const { useParticipants, useLocalParticipant } = useCallStateHooks();
  const participants = useParticipants();
  const localParticipant = useLocalParticipant();

  // Separate remote participants (agent) from local
  const remoteParticipants = participants.filter(
    (p) => p.sessionId !== localParticipant?.sessionId
  );

  return (
    <div className="call-ui">
      {/* Main video area */}
      <div className="video-grid">
        {/* Local camera (user) */}
        {localParticipant && (
          <div className="video-tile local">
            <ParticipantView
              participant={localParticipant}
              trackType="videoTrack"
            />
            <span className="video-label">You</span>
          </div>
        )}

        {/* Remote participants (agent video tracks) */}
        {remoteParticipants.map((p) => (
          <div key={p.sessionId} className="video-tile remote">
            <ParticipantView participant={p} trackType="videoTrack" />
            <span className="video-label">
              {p.name || p.userId || "WorldLens Agent"}
            </span>
          </div>
        ))}
      </div>

      {/* Play remote participants' audio (agent voice) */}
      <ParticipantsAudio />

      {/* Controls */}
      <div className="call-controls">
        <button className="btn btn-danger" onClick={onLeave}>
          Leave Call
        </button>
        <span className="participant-count">
          {participants.length} participant{participants.length !== 1 ? "s" : ""}
        </span>
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
  const [client, setClient] = useState<StreamVideoClient | null>(null);
  const [call, setCall] = useState<ReturnType<StreamVideoClient["call"]> | null>(null);
  const [joined, setJoined] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize Stream Video client
  useEffect(() => {
    if (!STREAM_API_KEY) {
      setError("VITE_STREAM_API_KEY not configured");
      return;
    }

    const videoClient = new StreamVideoClient({
      apiKey: STREAM_API_KEY,
      user: { id: userId, name: userName },
      tokenProvider: async () => {
        // For development: use the backend to generate a user token
        // In production, this should come from your auth server
        try {
          const res = await fetch(
            `${import.meta.env.VITE_BACKEND_URL || "http://localhost:8000"}/token?user_id=${userId}`
          );
          if (res.ok) {
            const data = await res.json();
            return data.token;
          }
        } catch (err) {
          console.warn("Token fetch failed, falling back to guest mode:", err);
          // Fall through to guest token
        }
        // Guest mode fallback (works for development)
        return "";
      },
    });

    setClient(videoClient);

    return () => {
      videoClient.disconnectUser();
    };
  }, [userId, userName]);

  // Join call when client is ready
  useEffect(() => {
    if (!client || !callId) return;
    let cancelled = false;

    const streamCall = client.call(callType, callId);
    setCall(streamCall);

    // The backend agent creates the call via POST /sessions.
    // We join without create — but retry a few times in case the agent
    // hasn't finished creating/joining the call yet (race condition).
    const tryJoin = async (attempts = 8, delayMs = 1500) => {
      for (let i = 0; i < attempts; i++) {
        if (cancelled) return;
        try {
          await streamCall.join({ create: false });
          if (cancelled) return;
          setJoined(true);
          streamCall.camera.enable();
          streamCall.microphone.enable();
          return;
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : String(err);
          // "call not found" means the agent hasn't created it yet — retry
          if (i < attempts - 1 && (msg.includes("not found") || msg.includes("404") || msg.includes("does not exist"))) {
            console.log(`Call not ready yet, retrying in ${delayMs}ms… (${i + 1}/${attempts})`);
            await new Promise((r) => setTimeout(r, delayMs));
            continue;
          }
          // Try once with create as final fallback
          try {
            await streamCall.join({ create: true });
            if (cancelled) return;
            setJoined(true);
            streamCall.camera.enable();
            streamCall.microphone.enable();
            return;
          } catch (fallbackErr: unknown) {
            console.error("Failed to join call (with create fallback):", fallbackErr);
            setError(`Failed to join call: ${fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr)}`);
          }
        }
      }
    };

    tryJoin();

    return () => {
      cancelled = true;
      streamCall.leave().catch(console.error);
    };
  }, [client, callId, callType]);

  const handleLeave = useCallback(async () => {
    if (call) {
      await call.leave();
      setJoined(false);
    }
    onLeave?.();
  }, [call, onLeave]);

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
    <StreamVideo client={client}>
      <StreamTheme>
        <StreamCall call={call}>
          <CallUI onLeave={handleLeave} />
        </StreamCall>
      </StreamTheme>
    </StreamVideo>
  );
}
