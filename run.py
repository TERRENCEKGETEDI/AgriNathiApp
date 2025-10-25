#!/usr/bin/env python3
"""
AgriNathi - Agricultural Voice Assistant
Main application runner
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("Starting AgriNathi Agricultural Voice Assistant...")
    print(f"Server will be available at: http://localhost:{port}")
    print("Voice recognition ready for isiZulu input")
    print("Mobile app available for Android/iOS")
    app.run(debug=False, host='0.0.0.0', port=port)