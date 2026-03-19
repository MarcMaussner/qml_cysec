import requests
import sys
import os

def render_dot(dot_file, output_file):
    if not os.path.exists(dot_file):
        print(f"Error: {dot_file} not found.")
        return

    with open(dot_file, 'r') as f:
        dot_content = f.read()

    url = 'https://quickchart.io/graphviz'
    params = {
        'graph': dot_content,
        'format': 'png'
    }

    print(f"Rendering {dot_file} to {output_file}...")
    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Success!")
    except Exception as e:
        print(f"Failed to render: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python render_dot.py <input.dot> <output.png>")
    else:
        render_dot(sys.argv[1], sys.argv[2])
