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
