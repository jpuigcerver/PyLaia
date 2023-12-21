#!/bin/sh -e
# Requires CI_PROJECT_DIR and CI_REGISTRY_IMAGE to be set.
# VERSION defaults to latest.
# Will automatically login to a registry if CI_REGISTRY, CI_REGISTRY_USER and CI_REGISTRY_PASSWORD are set.
# Will only push an image if $CI_REGISTRY is set.

if [ -z "$VERSION" ]; then
	VERSION=${CI_COMMIT_TAG:-latest}
fi

if [ -z "$VERSION" ] || [ -z "$CI_PROJECT_DIR" ] || [ -z "$CI_REGISTRY_IMAGE" ]; then
	echo Missing environment variables
	exit 1
fi

if [ -n "$CI_REGISTRY" ] && [ -n "$CI_REGISTRY_USER" ] && [ -n "$CI_REGISTRY_PASSWORD" ]; then
	echo Logging in to container registry…
	echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin "$CI_REGISTRY"
fi

IMAGE_TAG="$CI_REGISTRY_IMAGE:$VERSION"

cd "$CI_PROJECT_DIR"
docker build -f Dockerfile . -t "$IMAGE_TAG"
if [ "$CI_COMMIT_REF_NAME" = "$CI_DEFAULT_BRANCH" ] || [ -n "$CI_COMMIT_TAG" ]; then
	docker push "$IMAGE_TAG"
fi
