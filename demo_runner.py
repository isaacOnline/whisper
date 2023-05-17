import whisper



model = whisper.load_model("tiny", device="cpu")
audio_path = r"<local file path>.mp3"
result = model.transcribe(audio_path)
text_result = result["text"]
print(text_result)
segments = result['segments']
for seg in segments:
    encoder_embeddings = seg["encoder_embeddings"]
    decoder_embeddings = seg["decoder_embeddings"]
    print(encoder_embeddings.shape, decoder_embeddings.shape)
    break