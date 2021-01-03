"""This program demonstrates what triple-quoted first line of the program does."""

# T. Harris 3-Jan-2021 created

# This program does nothing, but if I ask for its help string, I'll get the above triple-quoted text.
# Based on this Stack Overflow answer: https://stackoverflow.com/a/47495442

# imports

import sys

# main program

if __name__=='__main__':
 if len(sys.argv)==2 and sys.argv[1]=='--help':
    print(__doc__)

# run this program from the command line with
# python triple-quote-help-demo --help

