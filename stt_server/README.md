# Robot Savo — STT + LLM Gateway (`stt_server`)

This service is the **speech front door** for Robot Savo running on the **PC/Mac** side.

From the Pi’s point of view the flow is:

```text
Human → Pi (ReSpeaker mic, audio)
Pi → Robot_Savo_Server /speech  (port 9000)

Robot_Savo_Server (on PC/Mac, inside Docker):
    stt_server:
        - decode audio
        - STT (faster-whisper) → transcript
        - call llm_server /chat → reply_text + intent + nav_goal
        - return combined JSON to Pi

Pi:
    - TTS (Piper) speaks reply_text
    - Nav stack uses intent + nav_goal

Pi only needs to know one HTTP endpoint:

http://PC-IP:9000/speech

