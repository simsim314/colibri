.PHONY: all glm gguf-inspect sgguf-convert sgguf-inspect portable test check cuda-test cuda-sgguf-test clean install uninstall

all glm gguf-inspect sgguf-convert sgguf-inspect portable test check cuda-test cuda-sgguf-test clean install uninstall:
	$(MAKE) -C c $@