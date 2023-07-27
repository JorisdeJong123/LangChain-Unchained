from pydub import AudioSegment
import io
import openai

# Function to get response from OpenAI Whisper API.
def transcribe_audio(audio_bytes, openai_api_key):
    # Get response from OpenAI Whisper API.
    response = openai.Audio.transcribe(
        api_key=openai_api_key,
        model="whisper-1",
        file=audio_bytes
    )
    
    return response

# Custom BytesIO class with a name attribute for Whisper
class NamedBytesIO(io.BytesIO):
    def __init__(self, content, name):
        super().__init__(content)
        self.name = name

def split_audio_into_chunks_variables(audio_file_path):
    # Load the audio file.
    meeting = AudioSegment.from_mp3(audio_file_path)
    length_file = meeting.duration_seconds

    # If duration is longer than 15 minutes, split the audio into 15-minute chunks.
    chunks = []
    if length_file > 900:
        chunks_count = int(length_file / 900)

        for i in range(chunks_count):
            start_time = i * 900 * 1000
            end_time = (i + 1) * 900 * 1000
            meeting_chunk = meeting[start_time:end_time]
            chunks.append(meeting_chunk)

        # If there is a remainder, split the remainder into a chunk.
        if length_file % 900 != 0:
            remainder = int(length_file % 900)
            start_time = chunks_count * 900 * 1000
            end_time = start_time + remainder * 1000
            meeting_chunk = meeting[start_time:end_time]
            chunks.append(meeting_chunk)

    return chunks