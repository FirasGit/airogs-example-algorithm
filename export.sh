#!/usr/bin/env bash

./build.sh

docker save airogs_algorithm | xz -T0 -c > airogs_algorithm.tar.xz
