def menu(prompt: str, options: list[str]) -> int:
    print(prompt)
    while True:
        for i, option in enumerate(options):
            print(str(i + 1), ". ", option, sep="")
        try:
            choice = int(input("> ").strip())
        except:
            continue

        if choice in range(1, len(options) + 1):
            return choice