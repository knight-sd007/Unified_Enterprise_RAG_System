pipeline {
    agent any

    parameters {
        string(name: 'DOCKERHUB_USERNAME', defaultValue: 'knightprime007', description: 'Docker Hub Registry Namespace')
        string(name: 'IMAGE_NAME', defaultValue: 'unified-enterprise-rag-system', description: 'Docker Image Repository Name')
        string(name: 'OCI_HOST', defaultValue: 'rag.vaikuntrix.in', description: 'Target Public Ingress Hostname')
    }

    environment {
        GIT_SHA = "${env.GIT_COMMIT ? env.GIT_COMMIT.take(7) : error('GIT_COMMIT is missing; immutable Git SHA tag is required')}"
        IMAGE_FULL_TAG = "${params.DOCKERHUB_USERNAME}/${params.IMAGE_NAME}:${env.GIT_SHA}"
        IMAGE_LATEST_TAG = "${params.DOCKERHUB_USERNAME}/${params.IMAGE_NAME}:latest"
        DOCKERHUB_CRED_ID = 'docker-hub-credentials'
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo "Checked out commit: ${env.GIT_COMMIT} (Short SHA: ${env.GIT_SHA})"
            }
        }

        stage('Secret Scan') {
            steps {
                script {
                    echo "Executing Gitleaks secret detection..."
                    sh 'docker run --rm -v "${WORKSPACE}:/source:ro" zricethezav/gitleaks:latest detect --source /source --verbose'
                }
            }
        }

        stage('Test & Quality Gates') {
            steps {
                script {
                    echo "Running automated test suite in isolated Python 3.12 container..."
                    sh '''
                        docker run --rm -v "${WORKSPACE}:/app" -w /app python:3.12-slim-bookworm sh -c "
                            pip install --no-cache-dir -r requirements.txt &&
                            pytest -v
                        "
                    '''
                }
            }
        }

        stage('Build ARM64 Image') {
            steps {
                script {
                    echo "Building multi-arch Docker container image: ${IMAGE_FULL_TAG}..."
                    sh "docker buildx build --platform linux/arm64 -t ${IMAGE_FULL_TAG} -t ${IMAGE_LATEST_TAG} --load ."
                }
            }
        }

        stage('Container Security Scan') {
            steps {
                script {
                    echo "Scanning container image ${IMAGE_FULL_TAG} with Trivy..."
                    sh "docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image --severity HIGH,CRITICAL --exit-code 1 ${IMAGE_FULL_TAG}"
                }
            }
        }

        stage('Push Docker Hub') {
            when { branch 'main' }
            steps {
                script {
                    echo "Publishing immutable container image ${IMAGE_FULL_TAG} to Docker Hub..."
                    withCredentials([usernamePassword(credentialsId: DOCKERHUB_CRED_ID, usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
                        sh 'echo "$DH_PASS" | docker login -u "$DH_USER" --password-stdin'
                        sh "docker push ${IMAGE_FULL_TAG}"
                        sh "docker push ${IMAGE_LATEST_TAG}"
                    }
                }
            }
        }

        stage('Deploy OCI') {
            when { branch 'main' }
            steps {
                script {
                    echo "Deploying P06 image ${IMAGE_FULL_TAG} to target host..."
                    sh """
                        if [ ! -f /opt/projects/enterprise-rag/.env ]; then
                            echo "ERROR: Production environment file /opt/projects/enterprise-rag/.env not found on deployment host!"
                            exit 1
                        fi
                        cp docker-compose.yml /opt/projects/enterprise-rag/docker-compose.yml
                        P06_IMAGE="${IMAGE_FULL_TAG}" docker compose --env-file /opt/projects/enterprise-rag/.env -f /opt/projects/enterprise-rag/docker-compose.yml pull
                        P06_IMAGE="${IMAGE_FULL_TAG}" docker compose --env-file /opt/projects/enterprise-rag/.env -f /opt/projects/enterprise-rag/docker-compose.yml up -d
                    """
                }
            }
        }

        stage('Post-Deployment Verification') {
            when { branch 'main' }
            steps {
                script {
                    sh """
                        MAX_ATTEMPTS=15
                        SLEEP_SECONDS=2
                        CURL_TIMEOUT=2
                        HEALTH_URL="http://127.0.0.1:8006/_stcore/health"

                        echo "Layer 1 Verification: Internal application readiness check (\$HEALTH_URL)..."

                        ATTEMPT=1
                        SUCCESS=0

                        while [ \$ATTEMPT -le \$MAX_ATTEMPTS ]; do
                            HTTP_CODE=\$(curl -s -o /dev/null -w "%{http_code}" --max-time "\$CURL_TIMEOUT" "\$HEALTH_URL") || HTTP_CODE="000"

                            if [ "\$HTTP_CODE" = "200" ]; then
                                echo "[Attempt \$ATTEMPT/\$MAX_ATTEMPTS] Layer 1 Healthcheck: PASS (HTTP 200 OK)"
                                SUCCESS=1
                                break
                            else
                                echo "[Attempt \$ATTEMPT/\$MAX_ATTEMPTS] Streamlit starting up (HTTP \$HTTP_CODE). Retrying..."
                                if [ "\$ATTEMPT" -lt "\$MAX_ATTEMPTS" ]; then
                                    sleep "\$SLEEP_SECONDS"
                                fi
                                ATTEMPT=\$((ATTEMPT + 1))
                            fi
                        done

                        if [ \$SUCCESS -ne 1 ]; then
                            echo "ERROR: Layer 1 readiness check failed after \$MAX_ATTEMPTS attempts against \$HEALTH_URL (Status: \$HTTP_CODE)!"
                            exit 1
                        fi

                        echo "Layer 1 Verification: Verifying running container image tag corresponds to ${IMAGE_FULL_TAG}..."
                        RUNNING_IMAGE=\$(docker inspect --format '{{.Config.Image}}' p06_rag_system 2>/dev/null || echo "NOT_RUNNING")
                        if [ "\$RUNNING_IMAGE" != "${IMAGE_FULL_TAG}" ]; then
                            echo "ERROR: Running container image (\$RUNNING_IMAGE) does not match deployed image ${IMAGE_FULL_TAG}!"
                            exit 1
                        fi
                        echo "Running container verified: \$RUNNING_IMAGE"
                    """

                    echo "Layer 2 Verification: Cloudflared Tunnel process verification..."
                    sh '''
                        if pgrep cloudflared >/dev/null || systemctl is-active cloudflared >/dev/null 2>&1 || docker ps | grep -q cloudflared; then
                            echo "Cloudflared tunnel status: RUNNING"
                        else
                            echo "ERROR: Cloudflared tunnel process is not active on host!"
                            exit 1
                        fi
                    '''

                    echo "Layer 3 Verification: Public HTTPS route (https://${params.OCI_HOST})..."
                    sh """
                        HTTP_STATUS=\$(curl -o /dev/null -s -w "%{http_code}" --max-time 10 https://${params.OCI_HOST} || echo "CURL_ERROR")
                        if [ "\$HTTP_STATUS" = "200" ]; then
                            echo "Public Cloudflare route: PASS (HTTP 200 OK)"
                        elif [ "\$HTTP_STATUS" = "503" ] || [ "\$HTTP_STATUS" = "403" ]; then
                            echo "Public Cloudflare route: CHALLENGED (Cloudflare Under Attack Mode Active — HTTP \$HTTP_STATUS. Deployment Healthy)."
                        else
                            echo "ERROR: Public Cloudflare route failed with status \$HTTP_STATUS"
                            exit 1
                        fi
                    """
                }
            }
        }

        stage('Target Disk Cleanup') {
            when { branch 'main' }
            steps {
                script {
                    echo "Performing disk cleanup for obsolete P06 images..."
                    sh """
                        docker image prune -f
                        docker container prune -f
                        docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep '${params.DOCKERHUB_USERNAME}/${params.IMAGE_NAME}' | grep -v '${env.GIT_SHA}' | grep -v 'latest' | awk '{print \$2}' | xargs -r docker rmi -f || true
                    """
                }
            }
        }
    }

    post {
        always {
            cleanWs(deleteDirs: true, notFailBuild: true)
        }
        success {
            echo "Successfully built, tested, scanned, published, and deployed P06 commit ${env.GIT_SHA}!"
        }
        failure {
            echo "Pipeline execution failed on commit ${env.GIT_COMMIT}."
        }
    }
}
