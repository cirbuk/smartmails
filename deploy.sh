#!/bin/bash

set -e

if [ $# -eq 0 ] ; then
    version_code=20180713t174317
    echo "using default version code $version_code"
else
    version_code=$1
    echo "using version code - $version_code"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
echo "timestamp - $timestamp"
container_name=smartmails
service=smartmails

docker build -t ${container_name} .

docker tag ${container_name} us.gcr.io/raydio-test/smartmails:${timestamp}

docker push us.gcr.io/raydio-test/smartmails:${timestamp}

if [ "$version_code" == "beta" ] ; then
    promote=--no-promote
fi

gcloud app deploy --version ${version_code} --image-url us.gcr.io/raydio-test/${service}:${timestamp} --quiet $promote
