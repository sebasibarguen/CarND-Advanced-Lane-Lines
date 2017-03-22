def convert_file(file_path, format, output_path):
    cmd = [
        'convert',  # ImageMagick Convert
        '-',  # Read original picture from StdIn
        '{}:-'.format(format),  # Write thumbnail with `format` to StdOut
    ]

    with Image.open(file_path) as image:

        p = Popen(cmd, stdout=PIPE, stdin=PIPE)
        img = p.communicate(input=image)[0]
        img = Image.open(cStringIO.StringIO(img))
        img.save(output_path)
