# WorldLens: System Architecture & Data Flow

```mermaid
flowchart LR
    subgraph Edge Device
        A[M5StickC Plus Camera (RTSP/MJPEG)]
        AA[On-device Buffer/Cache]
    end
    subgraph Backend Relay
        B[Python Relay Script (m5_bridge.py)]
        BB[Frame Queue]
    end
    subgraph Stream Network
        C[Stream Edge Network (WebRTC)]
        CC[Session Metadata Store]
    end
    subgraph WorldLens Backend
        D[Vision Agents Python Backend]
        D1[Session Manager]
        D2[Local Storage (JSON/SQLite)]
        D3[Frame Store (Local)]
        D4[Event Bus (In-Memory)]
        D5[Processor Orchestrator]
        D6[Analytics Engine]
    end
    subgraph Processors
        E1[YOLOPoseProcessor (SignBridge)]
        E2[YOLO11 Detection (GuideLens)]
        E3[Multi-VLM OCR (Gemini/Grok/Azure/NVIDIA/HF)]
        E4[HuggingFace NLP]
        E5[NVIDIA NIM VLM]
    end
    subgraph Agentic Reasoning
        F1[Gemini 2.5 Flash Realtime]
        F2[MCP Tools]
        F3[Google Maps API]
        F4[Spatial Memory DB (SQLite)]
    end
    subgraph Frontend
        G[React 19 (Vite 7)]
        G1[WebRTC Client]
        G2[User Session Store (IndexedDB)]
        G3[3D Avatar (React Three Fiber)]
        G4[AlertOverlay (Mock Haptics)]
        G5[Telemetry Panel]
        G6[OCR Overlay]
        G7[Chat/History Log]
    end
    %% Data Flows
    A --> AA
    AA --> B
    B --> BB
    BB --> C
    C --> CC
    C --> D
    D1 --> D2
    D1 --> D3
    D1 --> D4
    D --> D5
    D5 --> E1
    D5 --> E2
    D5 --> E3
    D5 --> E4
    D5 --> E5
    E1 -->|Skeletal Data| E4
    E2 -->|Detections| E3
    E3 -->|OCR Results| D6
    E4 -->|NLP Output| D6
    E5 -->|Scene Analysis| D6
    D6 -->|Insights| F1
    D6 -->|Tool Calls| F2
    F2 --> F3
    F2 --> F4
    F1 -->|Synthesized Audio/JSON| C
    C --> G1
    G1 --> G
    G --> G2
    G --> G3
    G --> G4
    G --> G5
    G --> G6
    G --> G7
    %% Data Storage
    D3 -.->|Frame Snapshots| D4
    D4 -.->|Event History| F4
    G2 -.->|Session Data| G7
    G7 -.->|User Feedback| D2
    style D2 fill:#f9f,stroke:#333,stroke-width:2px
    style D3 fill:#bbf,stroke:#333,stroke-width:2px
    style D4 fill:#bfb,stroke:#333,stroke-width:2px
    style F4 fill:#ffb,stroke:#333,stroke-width:2px
    style G2 fill:#fcc,stroke:#333,stroke-width:2px
    style G7 fill:#eee,stroke:#333,stroke-width:2px
    style D6 fill:#cfc,stroke:#333,stroke-width:2px
    style F1 fill:#ccf,stroke:#333,stroke-width:2px
    style F2 fill:#fcf,stroke:#333,stroke-width:2px
    style F3 fill:#ffc,stroke:#333,stroke-width:2px
    style E5 fill:#cff,stroke:#333,stroke-width:2px
    style G3 fill:#ffb,stroke:#333,stroke-width:2px
    style G4 fill:#fcc,stroke:#333,stroke-width:2px
    style G5 fill:#eee,stroke:#333,stroke-width:2px
    style G6 fill:#bfb,stroke:#333,stroke-width:2px
    style AA fill:#eee,stroke:#333,stroke-width:2px
    style BB fill:#eee,stroke:#333,stroke-width:2px
    style CC fill:#eee,stroke:#333,stroke-width:2px
    style D1 fill:#cfc,stroke:#333,stroke-width:2px
    style D5 fill:#cfc,stroke:#333,stroke-width:2px
    style E1 fill:#cfc,stroke:#333,stroke-width:2px
    style E2 fill:#cfc,stroke:#333,stroke-width:2px
    style E3 fill:#cfc,stroke:#333,stroke-width:2px
    style E4 fill:#cfc,stroke:#333,stroke-width:2px
    style G1 fill:#cfc,stroke:#333,stroke-width:2px
```