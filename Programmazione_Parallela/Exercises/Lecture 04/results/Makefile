CC = gcc
HEADS = ${wildcard *.h}
LIBDIR = lib
DYN_EXT = so
SHARED = static
ifeq "${SHARED}" "static"
CFLAGS = -O3 -Wall -pedantic -std=c17
else
CFLAGS = -fPIC -O3 -Wall -pedantic -std=c17
endif

all: main

ifeq "${SHARED}" "static"
main: main.o ${LIBDIR}/libbst.a
	${CC} ${CFLAGS} -o $@ $< -L./${LIBDIR} -lbst
else
main: main.o ${LIBDIR}/libbst.${DYN_EXT}
	${CC} ${CFLAGS} -o $@ $< -L./${LIBDIR} -lbst
endif

${LIBDIR}/libbst.a: bst.o tnode.o
	mkdir -p ${LIBDIR}
	ar rs $@ $^

${LIBDIR}/libbst.${DYN_EXT}: bst.o tnode.o
	mkdir -p ${LIBDIR}
	${CC} ${CFLAGS} -o $@ -shared $^

%.o: %.c ${HEADS}
	${CC} ${CFLAGS} -c $<

.PHONY: clean

clean:
	rm -f *.o
	rm -rf ${LIBDIR}
	rm -f main