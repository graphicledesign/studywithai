import streamlit as st
import webbrowser

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def main():
    st.title("StudyTech")

    # Add back arrow button in the sidebar
    if st.sidebar.button("← Back to Homepage"):
        webbrowser.open_new_tab("https://homepage-ui-psi.vercel.app/")

    # Define icons for each page
    page_icons = {
        "Chat with Websites": "🌐",
        "Generate Notes from YouTube Video": "📄",
        "Chat with Textbooks": "📚"
    }

    # Define buttons with icons and page names
    button_texts = {
        "Chat with Websites": "Chat with Websites",
        "Generate Notes from YouTube Video": "Generate Notes",
        "Chat with Textbooks": "Chat with Textbooks"
    }

    # Define links for each button
    button_links = {
        "Chat with Websites": "https://studytechgenerativetools.streamlit.app/Chat_With_Websites/",
        "Generate Notes from YouTube Video": "https://studytechgenerativetools.streamlit.app/Video_Summary",
        "Chat with Textbooks": "https://studytechgenerativetools.streamlit.app/Chat_With_Textbooks"
    }

    # Display sections with buttons and paragraphs
    for page_name, button_text in button_texts.items():
        st.markdown(
            f"""
            <div style="background-color: #F1F2F6; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h2 style="margin-bottom: 5px;"><span style="font-size: 24px; margin-right: 10px;">{page_icons[page_name]}</span>{page_name}</h2>
                <p>Meet Julia, your AI companion! She's here to chat about anything and everything. With a vast amount of knowledge at her disposal, Julia is always ready to answer your questions, offer advice, and share her insights.</p>
                <a href="{button_links[page_name]}" class="button-{page_name.replace(" ", "-")}" target="_blank">{button_text}</a>
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )

        # Apply CSS for button styles
        st.markdown(
            f"""
            <style>
                .button-{page_name.replace(" ", "-")} {{
                    display: inline-block;
                    background-color: transparent;
                    color: black;
                    padding: 10px 20px;
                    border: 2px solid black;
                    border-radius: 8px;
                    text-decoration: none; /* Remove underline for anchor tags */
                    cursor: pointer;
                    font-size: 16px;
                    transition: background-color 0.3s, color 0.3s;
                }}
                .button-{page_name.replace(" ", "-")}:hover,
                .button-{page_name.replace(" ", "-")}:active {{
                    background-color: black;
                    color: white;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
