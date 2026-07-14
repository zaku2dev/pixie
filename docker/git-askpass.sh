#!/bin/sh
# Feeds the GitHub token to git over HTTPS without persisting it on disk.
# git invokes this (via the GIT_ASKPASS env var) whenever it needs the password
# for the https://x-access-token@github.com/... remote. The username is already
# in the remote URL, so git only asks for the password — which is the token.
#
# Inject the token at container launch, e.g.  -e GITHUB_TOKEN=ghp_xxx
# then just run `git pull` inside the container.
printf '%s\n' "${GITHUB_TOKEN}"
