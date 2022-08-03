import os
import re
from data.ai2_objects import ALL_AI2THOR_OBJECT_CLASSES


def main():
    # read data from file and combine into a gigantic string
    base_dir = "/data/anthony/wikihow"
    files = os.listdir(os.path.join(base_dir, "articles"))

    keywords = ALL_AI2THOR_OBJECT_CLASSES

    def filter_files(fn):
        for word in keywords:
            if word.lower() in fn.lower():
                return True
        return False

    files = list(filter(filter_files, files))

    text = ""

    print(f"wikihow data has {len(files)} files")
    for _, file in enumerate(files):
        # print(file)
        with open(os.path.join(base_dir, file), "r") as f:
            content = f.readlines()
            for i, sent in enumerate(content):
                if sent.strip() == "@summary":
                    instr = content[i + 1]
                    instr = re.sub(r"[^\w\s]", "", instr).strip()
                    text += instr + ". "
                elif sent.strip() == "@article":
                    # not using the full article right now
                    pass

    # save filtered and concatenated text
    with open(os.path.join(base_dir, "combined_text.txt"), "w") as f:
        f.write(text)

    return text


if __name__ == "__main__":
    main()
