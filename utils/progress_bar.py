def progress_bar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    progress = 'Progress: [%s%s] %d %%' % (arrow, spaces, percent)
    print('\r', 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='')
