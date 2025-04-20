import streamlit as st
from dotenv import load_dotenv
import os
import re
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

# Hide Streamlit style elements
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Prompt for summarization
prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here: """

# Function to extract video ID from various YouTube URL formats
def extract_video_id(youtube_url):
    # Regular expressions to match different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URLs
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})' # Standard watch URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return None

# Getting the transcript data from YouTube videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            return None, "Invalid YouTube URL. Please check the URL and try again."
        
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript, None
    except NoTranscriptFound:
        return None, "No transcript found for this video. The video might not have captions or subtitles available."
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

# Getting the summary based on Prompt from Google Gemini
def generate_gemini_content(transcript_text, prompt):
    try:
        # List available models to debug
        available_models = []
        for model in genai.list_models():
            if "gemini" in model.name.lower():
                available_models.append(model.name)
        
        if not available_models:
            return None, "No Gemini models available with your API key. Please check your API key and permissions."
        
        # Try to use the most appropriate Gemini model available
        model_to_use = None
        preferred_models = ["gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
        for preferred in preferred_models:
            for available in available_models:
                if preferred in available:
                    model_to_use = available
                    break
            if model_to_use:
                break
        
        if not model_to_use:
            model_to_use = available_models[0]  # Use first available if no preferred match
        
        st.info(f"Using model: {model_to_use}")
        
        # Generate the content with the selected model
        model = genai.GenerativeModel(model_to_use)
        
        # Ensure the input isn't too long
        max_chars = 30000  # Adjust based on model limits
        trimmed_text = transcript_text[:max_chars] if len(transcript_text) > max_chars else transcript_text
        
        response = model.generate_content(prompt + trimmed_text)
        return response.text, None
    except Exception as e:
        return None, f"Error generating summary: {str(e)}\n\nAvailable models: {available_models if 'available_models' in locals() else 'Unknown'}"

# Page title
st.title("📄 YouTube Transcript to Detailed Notes Converter")

# Input for YouTube link
youtube_link = st.text_input("Enter YouTube Video Link:")

# Display thumbnail if link is provided
if youtube_link:
    video_id = extract_video_id(youtube_link)
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
    else:
        st.error("Invalid YouTube URL. Please enter a valid YouTube video link.")

# Button to get notes
if st.button("Get Detailed Notes"):
    if not youtube_link:
        st.error("Please enter a YouTube video link.")
    else:
        # First check if API key is set
        if not api_key:
            st.error("Google API key not found. Please make sure you have set the GOOGLE_API_KEY in your .env file.")
        else:
            with st.spinner("Extracting transcript and generating notes..."):
                # Extract transcript
                transcript_text, error_msg = extract_transcript_details(youtube_link)
                
                if error_msg:
                    st.error(error_msg)
                elif transcript_text:
                    # Generate summary
                    summary, summary_error = generate_gemini_content(transcript_text, prompt)
                    
                    if summary_error:
                        st.error(summary_error)
                    else:
                        st.markdown("## Detailed Notes:")
                        st.write(summary)
                        
                        # Option to download the summary
                        st.download_button(
                            label="Download Notes",
                            data=summary,
                            file_name="youtube_summary.txt",
                            mime="text/plain"
                        )
