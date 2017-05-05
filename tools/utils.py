def ColorPrint(str, idx = 0):
  print ('\033[1;31m' if idx == 0 else '\033[1;32m'), str, '\033[0m'

  