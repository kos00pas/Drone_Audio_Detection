import re

def generate_toc(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    toc = []
    for line in content:
        if line.startswith('#'):
            header = line.strip()
            level = header.count('#')
            title = header.replace('#', '').strip()
            anchor = re.sub(r'[^a-zA-Z0-9\s]', '', title).replace(' ', '-').lower()
            toc.append(f"{'  ' * (level - 1)}- [{title}](#{anchor})")

    return "\n".join(toc)

if __name__ == "__main__":
    toc = generate_toc('README.md')
    print(toc)
