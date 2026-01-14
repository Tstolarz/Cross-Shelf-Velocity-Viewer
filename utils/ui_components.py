import streamlit as st
import time

def time_it(func_name):
    """Performance timing decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds

            # Store timing in session state for display
            if 'timing_data' not in st.session_state:
                st.session_state.timing_data = {}
            st.session_state.timing_data[func_name] = duration

            print(f"⏱️ {func_name}: {duration:.1f}ms")
            return result
        return wrapper
    return decorator

def apply_custom_css():
    """Apply custom CSS styling and keyboard navigation"""
    st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
    h1 {
        padding-top: 0rem;
        margin-top: 0rem;
    }
    /* Improve metric display responsiveness */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        overflow: hidden;
        min-width: 0;
    }
    [data-testid="metric-container"] > div {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    /* Responsive text sizing */
    @media (max-width: 768px) {
        [data-testid="metric-container"] {
            font-size: 0.9rem;
        }
    }

    /* Auto-fade notification styles */
    [data-testid="stSuccess"], [data-testid="stInfo"] {
        animation: autoFadeOut 5s ease-in-out forwards;
    }

    @keyframes autoFadeOut {
        0% { opacity: 1; visibility: visible; }
        80% { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; height: 0; margin: 0; padding: 0; overflow: hidden; }
    }

    /* Highlight clickable page links so the script names stand out */
    [data-testid="stSidebarNav"] {
        margin-bottom: 1rem;
        position: relative;
    }
    [data-testid="stSidebarNav"]::before {
        content: "Select a page:";
        display: block;
        font-size: 1.2rem;
        font-weight: 600;
        text-transform: capitalize;
        letter-spacing: 0.05em;
        font-family: "Source Sans Pro", var(--font-family, "Source Sans Pro"), sans-serif;
        color: rgba(255, 255, 255, 0.85);
        margin-bottom: 0.45rem;
    }
    [data-testid="stSidebarNav"] ul {
        padding: 0.35rem;
        border-radius: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(34, 255, 0, 0.35);
        box-shadow: 0 0 18px rgba(13, 112, 23, 0.8);
    }
    [data-testid="stSidebarNav"] a {
        display: flex;
        align-items: center;
        padding: 0.35rem 0.65rem;
        border-radius: 0.5rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        color: #fafafa !important;
        text-shadow: 0 0 4px rgba(0, 0, 0, 0.45);
        transition: transform 0.15s ease, background 0.15s ease, color 0.15s ease;
    }
    [data-testid="stSidebarNav"] a:hover,
    [data-testid="stSidebarNav"] a:focus {
        background: rgba(40, 255, 140, 0.2);
        transform: translateX(4px);
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(90deg, #ffc94b, #ff9f1c);
        color: #151515 !important;
        box-shadow: 0 0 15px rgba(255, 187, 68, 0.4);
        text-shadow: none;
    }
    nav[aria-label="Page navigation"],
    nav[aria-label="pages"] {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-bottom: 1rem;
    }
    nav[aria-label="Page navigation"]::before,
    nav[aria-label="pages"]::before {
        content: "Select a page";
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: "Source Sans Pro", var(--font-family, "Source Sans Pro"), sans-serif;
        color: rgba(255, 255, 255, 0.85);
    }
    nav[aria-label="Page navigation"] a,
    nav[aria-label="pages"] a {
        border-radius: 0.5rem;
        padding: 0.35rem 0.75rem;
        font-weight: 600;
    }
    nav[aria-label="Page navigation"] a[aria-current="page"],
    nav[aria-label="pages"] a[aria-current="page"] {
        background: linear-gradient(90deg, #ffc94b, #ff9f1c);
        color: #151515 !important;
        box-shadow: 0 0 15px rgba(255, 187, 68, 0.4);
    }
</style>

<script>
document.addEventListener('keydown', function(event) {
    // Only trigger on left or right arrow keys
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        // Prevent default browser behavior
        event.preventDefault();

        // Find the appropriate navigation buttons
        const leftButtons = document.querySelectorAll('[data-testid="baseButton-secondary"]:has([title="Previous time step"]), button[title="Previous time step"]');
        const rightButtons = document.querySelectorAll('[data-testid="baseButton-secondary"]:has([title="Next time step"]), button[title="Next time step"]');

        if (event.key === 'ArrowLeft') {
            // Click the left arrow button (previous time step)
            for (let button of leftButtons) {
                if (button.textContent.includes('◀') || button.title === 'Previous time step') {
                    button.click();
                    break;
                }
            }
        } else if (event.key === 'ArrowRight') {
            // Click the right arrow button (next time step)
            for (let button of rightButtons) {
                if (button.textContent.includes('▶') || button.title === 'Next time step') {
                    button.click();
                    break;
                }
            }
        }
    }
});
</script>
""", unsafe_allow_html=True)
