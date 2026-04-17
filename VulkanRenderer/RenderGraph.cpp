#include "RenderGraph.hpp"

using Gfx::RenderGraph;

RenderGraph::RenderGraph(const RHI& rhi): 
    m_rhi(rhi)
{
}

void RenderGraph::init()
{
    // allocate one command buffer per swapchain image (common simple approach)
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *m_rhi.getCommandPool();
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = m_rhi.getMaxFramesInFlight();

    // vk::raii::CommandBuffers returns a container of RAII CommandBuffer objects;
    // move them into our vector so we can index per image.
    vk::raii::CommandBuffers tempCmds{ m_rhi.getDevice(), allocInfo};
    m_commandBuffers.reserve(allocInfo.commandBufferCount);
    for (uint32_t i = 0; i < allocInfo.commandBufferCount; ++i)
    {
        m_commandBuffers.emplace_back(std::move(tempCmds[i]));
    }

    // create per-frame semaphores and fences
    m_presentCompleteSemaphores.clear();
    m_renderFinishedSemaphores.clear();
    m_inFlightFences.clear();

    for (uint32_t i = 0; i < allocInfo.commandBufferCount; ++i)
    {
        m_presentCompleteSemaphores.emplace_back(m_rhi.getDevice(), vk::SemaphoreCreateInfo{});
        m_renderFinishedSemaphores.emplace_back(m_rhi.getDevice(), vk::SemaphoreCreateInfo{});
        // start signaled so the first wait doesn't block forever if user forgets
        m_inFlightFences.emplace_back(m_rhi.getDevice(), vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
}

void RenderGraph::executeFrame()
{
    // Round-robin frame index for per-frame sync objects
    const uint32_t frameIndex = m_currentFrame % m_rhi.getMaxFramesInFlight();
    auto& inFlightFence = m_inFlightFences[frameIndex];
    auto& presentComplete = m_presentCompleteSemaphores[frameIndex];
    auto& renderFinished = m_renderFinishedSemaphores[frameIndex];
    auto& cmd = m_commandBuffers[frameIndex];

    // Wait for fence for this frame to be signaled (previous GPU work finished)
    m_rhi.getDevice().waitForFences(*inFlightFence, true, UINT64_MAX);

    // Acquire next image
    auto acquireResult = m_rhi.getSwapChain().acquireNextImage(UINT64_MAX, *presentComplete, nullptr);
    auto imageIndex = acquireResult.second;

    // reset command buffer for this image
    cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // For each pass, optionally insert an image layout transition, then call the user record callback.
    // We assume all passes render to the swapchain color image directly in this simple sample.
    for (auto& pass : m_passes)
    {
        std::vector<vk::ImageMemoryBarrier2> barriers{};

        for (auto& transitionInfo : pass.transitionInfos) 
        {
            if (transitionInfo.oldLayout != transitionInfo.newLayout) 
            {
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

        if (barriers.size()) 
        {
            vk::DependencyInfo dependencyInfo{};
            dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            dependencyInfo.pImageMemoryBarriers = barriers.data();
            cmd.pipelineBarrier2(dependencyInfo);
        }

        // Call the pass record function to record draw/compute commands.
        if (pass.recordFunc)
        {
            pass.recordFunc(cmd, imageIndex);
        }
    }

    cmd.end();

    // reset the fence to unsignaled before submit
    m_rhi.getDevice().resetFences(*inFlightFence);

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

    m_rhi.getGraphicsQueue().submit(submitInfo, *inFlightFence);

    // Present: wait on renderFinished
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &*renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &*m_rhi.getSwapChain();
    presentInfo.pImageIndices = &imageIndex;

    m_rhi.getPresentQueue().presentKHR(presentInfo);

    // Advance frame index
    ++m_currentFrame;
}
