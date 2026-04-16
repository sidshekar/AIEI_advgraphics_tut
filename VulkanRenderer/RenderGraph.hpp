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

#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <functional>
#include <string>
#include <stdexcept>

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

    std::vector<AttachmentTransitionInfo> transitionInfos;
};

class RenderGraph
{
public:
    // Construct with references to objects managed elsewhere (HelloTriangleApplication keeps lifetime)
    RenderGraph(vk::raii::Device& device,
        vk::raii::SwapchainKHR& swapchain,
        vk::raii::Queue& graphicsQueue,
        vk::raii::Queue& presentQueue,
        vk::raii::CommandPool& commandPool)
        : m_device(device),
        m_swapchain(swapchain),
        m_graphicsQueue(graphicsQueue),
        m_presentQueue(presentQueue),
        m_commandPool(commandPool),
        m_imageCount(swapchain.getImages().size()) {
    }

    // Add a render pass node. Nodes are executed in the order they are added.
    void addPass(RenderPassNode const& node) {
        m_passes.push_back(node);
    }

    // Initialize per-frame resources (command buffers, semaphores, fences).
    // Must be called after creating swapchain and image views.
    void init() {
        // allocate one command buffer per swapchain image (common simple approach)
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *m_commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = m_imageCount;

        // vk::raii::CommandBuffers returns a container of RAII CommandBuffer objects;
        // move them into our vector so we can index per image.
        vk::raii::CommandBuffers tempCmds{ m_device, allocInfo };
        m_commandBuffers.reserve(m_imageCount);
        for (uint32_t i = 0; i < m_imageCount; ++i) {
            m_commandBuffers.emplace_back(std::move(tempCmds[i]));
        }

        // create per-frame semaphores and fences
        m_presentCompleteSemaphores.clear();
        m_renderFinishedSemaphores.clear();
        m_inFlightFences.clear();

        for (uint32_t i = 0; i < m_imageCount; ++i) {
            m_presentCompleteSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo{});
            m_renderFinishedSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo{});
            // start signaled so the first wait doesn't block forever if user forgets
            m_inFlightFences.emplace_back(m_device, vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
        }
    }

    // Execute full frame: acquire, record each pass, submit, present.
    // This implementation uses a single submit of the full set of recorded command buffers
    // and the classic SubmitInfo with semaphores and a fence. Image transitions inside passes
    // use pipelineBarrier2 (ImageMemoryBarrier2 + DependencyInfo).
    void executeFrame() {
        // Round-robin frame index for per-frame sync objects
        const uint32_t frameIndex = m_currentFrame % m_imageCount;
        auto& inFlightFence = m_inFlightFences[frameIndex];
        auto& presentComplete = m_presentCompleteSemaphores[frameIndex];
        auto& renderFinished = m_renderFinishedSemaphores[frameIndex];
        auto& cmd = m_commandBuffers[frameIndex];

        // Wait for fence for this frame to be signaled (previous GPU work finished)
        m_device.waitForFences(*inFlightFence, true, UINT64_MAX);

        // Acquire next image
        auto acquireResult = m_swapchain.acquireNextImage(UINT64_MAX, *presentComplete, nullptr);
        auto imageIndex = acquireResult.second;

        // reset command buffer for this image
        cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

        // For each pass, optionally insert an image layout transition, then call the user record callback.
        // We assume all passes render to the swapchain color image directly in this simple sample.
        for (auto const& pass : m_passes) {
            std::vector<vk::ImageMemoryBarrier2> barriers{};

            for (auto const& transitionInfo : pass.transitionInfos) {
                if (transitionInfo.oldLayout != transitionInfo.newLayout) {
                    vk::ImageMemoryBarrier2 barrier{};
                    barrier.srcStageMask = transitionInfo.srcStageMask;
                    barrier.srcAccessMask = transitionInfo.srcAccessMask;
                    barrier.dstStageMask = transitionInfo.dstStageMask;
                    barrier.dstAccessMask = transitionInfo.dstAccessMask;
                    barrier.oldLayout = transitionInfo.oldLayout;
                    barrier.newLayout = transitionInfo.newLayout;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.image = transitionInfo.images[frameIndex];
                    barrier.subresourceRange.aspectMask = transitionInfo.aspectMask;
                    barrier.subresourceRange.levelCount = 1;
                    barrier.subresourceRange.layerCount = 1;
                    barriers.emplace_back(std::move(barrier));
                }
            }

            if (barriers.size()) {
                vk::DependencyInfo dependencyInfo{};
                dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
                dependencyInfo.pImageMemoryBarriers = barriers.data();
                cmd.pipelineBarrier2(dependencyInfo);
            }

            // Call the pass record function to record draw/compute commands.
            if (pass.recordFunc) {
                pass.recordFunc(cmd, imageIndex);
            }
        }

        cmd.end();

        // reset the fence to unsignaled before submit
        m_device.resetFences(*inFlightFence);

        // Submit: wait on presentComplete, signal renderFinished
        vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        vk::SubmitInfo submitInfo{};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &*presentComplete;
        submitInfo.pWaitDstStageMask = &waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &*renderFinished;

        m_graphicsQueue.submit(submitInfo, *inFlightFence);

        // Present: wait on renderFinished
        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &*renderFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &*m_swapchain;
        presentInfo.pImageIndices = &imageIndex;

        m_presentQueue.presentKHR(presentInfo);

        // Advance frame index
        ++m_currentFrame;
    }

private:
    const vk::raii::Device& m_device;
    const vk::raii::SwapchainKHR& m_swapchain;
    const vk::raii::Queue& m_graphicsQueue;
    const vk::raii::Queue& m_presentQueue;
    const vk::raii::CommandPool& m_commandPool;
    const uint32_t m_imageCount;

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