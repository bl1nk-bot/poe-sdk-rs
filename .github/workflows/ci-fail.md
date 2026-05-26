---
description: |
  Workflow triggered when CI fails to label the PR and add a comment.

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches:
      - main
      - develop

permissions:
  contents: read
  issues: read
  pull-requests: read

safe-outputs:
  add-labels:
  add-comment:

tools:
  github:
    toolsets: [issues, pull_requests]
    min-integrity: none

timeout-minutes: 10
---

# CI Failure Handler

This workflow runs when the `CI` workflow completes on the specified branches. Your job is to check if it failed, and if so, find the associated pull requests, add the label "fix ci", and add a comment saying "@Type to search workspace files...".

## Instructions

1. Check the conclusion of the workflow run that triggered this event.
   - The conclusion is available in `${{ github.event.workflow_run.conclusion }}`.
   - If the conclusion is NOT "failure", stop and do nothing.

2. If the conclusion is "failure":
   - The head SHA of the commit that triggered the workflow run is `${{ github.event.workflow_run.head_sha }}`.
   - The run ID is `${{ github.event.workflow_run.id }}`.
   - Find any pull requests associated with this head SHA or run ID.
   - For each associated pull request:
     - Add the label "fix ci" to the pull request.
     - Add a comment to the pull request with the text: `@Type to search workspace files...`
