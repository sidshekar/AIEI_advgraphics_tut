// RenderGraph.hpp
//
// - Encapsulates acquire -> record -> submit -> present flow
// - Manages per-frame semaphores and fences
// - Demonstrates image layout transitions using synchronization2 (pipelineBarrier2 / ImageMemoryBarrier2)
// - Provides a minimal "pass" API: each pass supplies a record callback that is called with the per-frame command buffer
//
// Usage sketch:
//   RenderGraph rg(device, swapChain, graphicsQueue, presentQueue, commandPool, swapChainImageViews, swapChainExtent);
//   rg.addPass("GeometryPass", [](vk::raii::CommandBuffer &cmd, uint32_t imageIndex){ /* record drawing */ }, 
//              /*transition*/ vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
//              /*src/dst access/stage*/ {}, vk::AccessFlagBits2::eColorAttachmentWrite,
//              vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
//   rg.init(); // allocates command buffers and sync objects
//   // each frame:
//     rg.executeFrame();
//
//
// Note: this code focuses on synchronization and orchestration. It intentionally avoids higher-level resource/lifetime
// management (frame graphs with automatic aliasing, barriers across many resources, queue ownership transfers, etc.).

#pragma once

#include <functional>

#include "RHI.hpp"

namespace Gfx
{
    struct RenderPassNode
    {
        // Human-readable name (for students / debugging)
        std::string name;

        // Record callback: receives RAII command buffer and acquired image index
        // The callback must record commands into the provided command buffer.
        using RecordFunc = std::function<void(vk::raii::CommandBuffer&, uint32_t)>;
        RecordFunc recordFunc;

        struct AttachmentTransitionInfo
        {
            std::vector<vk::Image> images; // images to transition (e.g. swapchain image for color, depth image for depth)
            vk::ImageAspectFlagBits aspectMask; // aspect of the image to transition (e.g. color, depth, stencil)

            // Simple image-layout transition requirements for the attachments used by the pass.
            // If no transition is needed, set oldLayout == newLayout.
            vk::ImageLayout oldLayout = vk::ImageLayout::eUndefined;
            vk::ImageLayout newLayout = vk::ImageLayout::eUndefined;

            // Access & stage masks for the barrier that moves image from oldLayout->newLayout
            vk::AccessFlags2 srcAccessMask = {};
            vk::AccessFlags2 dstAccessMask = {};
            vk::PipelineStageFlags2 srcStageMask = {};
            vk::PipelineStageFlags2 dstStageMask = {};
        };

        struct BufferTransitionInfo
        {
            std::vector<vk::Buffer> buffers; // buffers to transition (e.g. uniform buffer, storage buffer)

            vk::AccessFlags2 srcAccessMask = {};
            vk::AccessFlags2 dstAccessMask = {};
            vk::PipelineStageFlags2 srcStageMask = {};
            vk::PipelineStageFlags2 dstStageMask = {};
		};

        std::vector<AttachmentTransitionInfo> attachmentInfos;
        std::vector<BufferTransitionInfo> bufferInfos;
    };

    class RenderGraph
    {
    public:
        // Construct with references to objects managed elsewhere (HelloTriangleApplication keeps lifetime)
        RenderGraph(const RHI& rhi);
        RenderGraph(const RenderGraph&) = delete;

        // Add a render pass node. Nodes are executed in the order they are added.
        void addPass(const RenderPassNode& node) { m_passes.push_back(node); }

        // Initialize per-frame resources (command buffers, semaphores, fences).
        // Must be called after creating swapchain and image views.
        void init();

        // Execute full frame: acquire, record each pass, submit, present.
        // This implementation uses a single submit of the full set of recorded command buffers
        // and the classic SubmitInfo with semaphores and a fence. Image transitions inside passes
        // use pipelineBarrier2 (ImageMemoryBarrier2 + DependencyInfo).
        void executeFrame();

    private:
		const RHI& m_rhi;

        // recorded passes
        std::vector<RenderPassNode> m_passes;

        // per-swapchain-image command buffers (RAII)
        std::vector<vk::raii::CommandBuffer> m_commandBuffers;

        // per-frame synchronization objects
        std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::raii::Fence> m_inFlightFences;

        uint64_t m_currentFrame = 0;
    };
}