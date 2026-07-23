.PHONY: all glm gguf-inspect portable test check cuda-test clean install uninstall

all glm gguf-inspect portable test check cuda-test clean install uninstall:
	$(MAKE) -C c $@