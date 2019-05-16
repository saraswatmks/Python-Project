#################################################################################
# Clean	                                                                        #
#################################################################################

clean:
	rm -rf .pytest_cache
	rm -rf vision.egg-info
	rm -rf target

#################################################################################
# Run test	                                                                    #
#################################################################################

test: clean
	python -m unittest tests.test_vision.TestVision


#################################################################################
# Build pex                                                                     #
#################################################################################

build: clean
	python --version
	pip install pex
	mkdir -p target
	pex . -v --disable-cache -r requirements_train.txt -R vision/docs -o target/vision_train.pex
	# pex . -v --disable-cache -r requirements_app.txt -R vision/docs -e vision.app.vision_app:app -o target/vision_app.pex

#################################################################################
# DOCKER                                                                        #
#################################################################################

.PHONY: docker-build docker-push-tag

## Build vision docker image to batch registry
docker-build:
	@echo "Building docker image"
	docker build . -t ${CI_REGISTRY_IMAGE}:$${CI_COMMIT_SHA:0:8}
	@echo "Pushing docker image to local registry"
	docker push ${CI_REGISTRY_IMAGE}:$${CI_COMMIT_SHA:0:8}

docker-push-tag:
	@echo "Pulling from gitlab registry"
	docker pull ${CI_REGISTRY_IMAGE}:$${CI_COMMIT_SHA:0:8}
	@echo "Pushing to gitlab registry"
	docker tag ${CI_REGISTRY_IMAGE}:$${CI_COMMIT_SHA:0:8} ${CI_REGISTRY_IMAGE}:${DOCKER_TAG}
	docker push ${CI_REGISTRY_IMAGE}:${DOCKER_TAG}
	@echo "Pushing to ECR"
	docker tag ${CI_REGISTRY_IMAGE}:$${CI_COMMIT_SHA:0:8} ${ECR_REGISTRY}:${DOCKER_TAG}
	docker push ${ECR_REGISTRY}:${DOCKER_TAG}

