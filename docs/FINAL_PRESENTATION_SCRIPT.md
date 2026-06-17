# Final Presentation Script: 3-Minute Live Demo

## Title

Digital Human Chat: A Multimodal Foundation Model Assistant

## 0:00–0:20 Opening

Hello, my final project is a digital human chat system.

The goal is to create an interactive assistant that can understand a user message, generate a natural response, synthesize a custom voice, and optionally produce a talking-avatar video.

This demo uses three foundation models: Qwen for language generation, Boson Higgs Audio for speech synthesis, and MuseTalk for talking-avatar video generation.

## 0:20–0:50 Interface Introduction

This is the live demo interface.

On the left side, I can type a message and see the conversation.

On the right side, I upload an avatar image. The avatar stays visible during the conversation.

There are two modes: Voice Mode and Video Mode.

## 0:50–1:30 Voice Mode Demo

First, I will demonstrate Voice Mode.

I type:

Hi, can you briefly introduce this final project?

The system sends the message to the local Qwen language model. Then Boson TTS generates speech using a custom voice.

This mode is faster and is useful when only voice interaction is needed.

## 1:30–2:30 Video Mode Demo

Next, I switch to Video Mode.

I type:

Can you explain what foundation models are used in this demo?

Now the system first generates the text response, then converts it into speech, and finally uses MuseTalk to generate a talking-avatar video.

The right-side digital human changes from idle or thinking state to speaking state.

## 2:30–3:00 Summary

The main result is an end-to-end multimodal digital human pipeline.

It combines language generation, voice synthesis, and audio-driven video generation in one live interface.

Thank you.
