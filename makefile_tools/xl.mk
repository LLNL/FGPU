XL_COMPILER_BIN_PATH:=$(shell which xlf)
XL_COMPILER_LIBS= $(subst bin/xlf,alllibs,$(XL_COMPILER_BIN_PATH))
