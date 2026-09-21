# Reference voices

Drop reference clips here. The file stem becomes the `voice_id`:

    voices/ishan_calm.wav      ->  "voice": "ishan_calm"
    voices/ishan_excited.wav   ->  "voice": "ishan_excited"

Guidelines for a good clone:
- 6-15 seconds, one speaker, no music / background noise
- natural sentences in the emotional register you want (calm, upbeat, ...)
- wav/flac/mp3, any sample rate (resampled internally)

Optional `voices.json` next to the clips adds metadata:

    {
      "ishan_calm": {"description": "Neutral support tone", "language": "en", "exaggeration": 0.4},
      "ishan_hindi": {"description": "Hindi, warm", "language": "hi"}
    }

Speaker conditionals are cached under `voices/.cache/` so restarts are instant.
