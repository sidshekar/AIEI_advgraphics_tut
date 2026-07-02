#include "RenderGraph.hpp"

#include <unordered_map>
#include <unordered_set>

using Gfx::ComputePipelineBuilder;
using Gfx::GraphicsPipelineBuilder;
using Gfx::RenderGraph;

ComputePipelineBuilder& ComputePipelineBuilder::shaderBinding(const Gfx::Buffer& buffer)
{
    auto index = static_cast<uint32_t>(m_pipelineCreateInfo.descriptorSetLayoutBindings.size());
    const auto& createInfo = buffer.getCreateInfo();

    auto descriptorType =
        (createInfo.usage & vk::BufferUsageFlagBits::eStorageBuffer) ?
        vk::DescriptorType::eStorageBuffer :
        vk::DescriptorType::eUniformBuffer;

    m_pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
        index,
        descriptorType,
        1,
        vk::ShaderStageFlagBits::eCompute,
        nullptr);

    DescriptorBinding descriptorBinding{};

    std::vector<vk::DescriptorBufferInfo> resourceInfos{};

    if (descriptorType == vk::DescriptorType::eUniformBuffer)
    {
        for (int i = 0; i < buffer.getBufferCount(); i++)
        {
            vk::DescriptorBufferInfo resourceInfo = {
                buffer.getBuffer(i),
                0,
                createInfo.size,
            };
            resourceInfos.emplace_back(std::move(resourceInfo));
        }
    }
    else
    {
        resourceInfos = { {
            buffer.getBuffer(0),
            0,
            createInfo.size,
        } };
    }

    descriptorBinding.type = descriptorType;
    descriptorBinding.data = resourceInfos;

    m_descriptorBindings.emplace_back(std::move(descriptorBinding));

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::shaderBinding(
    std::variant<const std::vector<Gfx::Image>*, const Gfx::Image*> images, 
    vk::ShaderStageFlagBits stage, 
    const Gfx::Sampler& sampler)
{
    auto index = static_cast<uint32_t>(m_pipelineCreateInfo.descriptorSetLayoutBindings.size());

    auto descriptorType =
        sampler.getSampler() != nullptr ?
        vk::DescriptorType::eCombinedImageSampler :
        vk::DescriptorType::eSampledImage;
    auto descriptorCount =
        std::holds_alternative<const std::vector<Image>*>(images) ?
        std::get<const std::vector<Image>*>(images)->size() :
        1;

    m_pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
        index,
        descriptorType,
        static_cast<uint32_t>(descriptorCount),
        stage,
        nullptr);

    DescriptorBinding descriptorBinding{};

    std::vector<std::vector<vk::DescriptorImageInfo>> resourceInfos{};

    if (descriptorCount == 1)
    {
        const auto& image = *std::get<const Image*>(images);

        m_shaderReadImages.emplace_back(std::move(image.getInfo()));

        for (int i = 0; i < image.getImageCount(); i++)
        {
            vk::DescriptorImageInfo resourceInfo = {
                sampler.getSampler(),
                image.getImageView(i),
                vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            resourceInfos.emplace_back(std::vector<vk::DescriptorImageInfo>{ std::move(resourceInfo) });
        }
    }
    else
    {
        resourceInfos.resize(1);

        const auto& imageArray = *std::get<const std::vector<Image>*>(images);

        for (const auto& image : imageArray)
        {
            m_shaderReadImages.emplace_back(std::move(image.getInfo()));

            vk::DescriptorImageInfo resourceInfo = {
                sampler.getSampler(),
                image.getImageView(0),
                vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            resourceInfos[0].emplace_back(std::move(resourceInfo));
        }
    }

    descriptorBinding.type = descriptorType;
    descriptorBinding.data = resourceInfos;

    m_descriptorBindings.emplace_back(std::move(descriptorBinding));

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::shaderBinding(const Gfx::Buffer* buffer, vk::ShaderStageFlagBits stage)
{
    auto index = static_cast<uint32_t>(m_pipelineCreateInfo.descriptorSetLayoutBindings.size());
    const auto& createInfo = buffer->getCreateInfo();

    auto descriptorType =
        (createInfo.usage & vk::BufferUsageFlagBits::eStorageBuffer) ?
        vk::DescriptorType::eStorageBuffer :
        vk::DescriptorType::eUniformBuffer;

    m_pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
        index,
        descriptorType,
        1,
        stage,
        nullptr);

    DescriptorBinding descriptorBinding{};

    std::vector<vk::DescriptorBufferInfo> resourceInfos{};

    if (descriptorType == vk::DescriptorType::eUniformBuffer)
    {
        for (int i = 0; i < buffer->getBufferCount(); i++)
        {
            vk::DescriptorBufferInfo resourceInfo = {
                buffer->getBuffer(i),
                0,
                createInfo.size,
            };
            resourceInfos.emplace_back(std::move(resourceInfo));
        }
    }
    else
    {
        resourceInfos = { {
            buffer->getBuffer(0),
            0,
            createInfo.size,
        } };
    }

    descriptorBinding.type = descriptorType;
    descriptorBinding.data = resourceInfos;

    m_descriptorBindings.emplace_back(std::move(descriptorBinding));

    return *this;
}

void RenderGraph::init()
{
    for (const auto& builder : m_pipelineBuilders)
    {
        if (std::holds_alternative<ComputePipelineBuilder>(builder))
        {
            const auto& computeBuilder = std::get<ComputePipelineBuilder>(builder);

            auto pipeline = m_rhi.createPipeline(computeBuilder.m_pipelineCreateInfo);
            auto descriptorSets = m_rhi.createDescriptorSets(pipeline.getDescriptorSetLayout(), computeBuilder.m_descriptorBindings);

            m_renderPasses.emplace_back(ComputePass{
                std::move(computeBuilder.m_name),
                computeBuilder.m_minDispatchThreadCount,
                std::move(pipeline),
                std::move(descriptorSets),
            });

            continue;
        }

        const auto& graphicsBuilder = std::get<GraphicsPipelineBuilder>(builder);

        auto pipeline = m_rhi.createPipeline(graphicsBuilder.m_pipelineCreateInfo);
        auto descriptorSets = m_rhi.createDescriptorSets(pipeline.getDescriptorSetLayout(), graphicsBuilder.m_descriptorBindings);

        m_renderPasses.emplace_back(GraphicsPass{
            std::move(graphicsBuilder.m_name),
            std::move(pipeline),
            std::move(descriptorSets),
            std::move(graphicsBuilder.m_vertexBuffer),
            std::move(graphicsBuilder.m_indexBuffer),
            std::move(graphicsBuilder.m_drawCommandBuffer),
            std::move(graphicsBuilder.m_colorTargetImages),
            std::move(graphicsBuilder.m_depthTargetImage),
            graphicsBuilder.m_usesSwapChainColor,
            graphicsBuilder.m_usesSwapChainDepth,
            std::move(graphicsBuilder.m_shaderReadImages),
        });
    }

    m_pipelineBuilders.clear();

    // allocate one command buffer per swapchain image (common simple approach)
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *m_rhi.getCommandPool();
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = m_rhi.getMaxFramesInFlight();

    // vk::raii::CommandBuffers returns a container of RAII CommandBuffer objects;
    // move them into our vector so we can index per image.
    vk::raii::CommandBuffers tempCmds{ m_rhi.getDevice(), allocInfo };
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
    // Auto-initialize if not done yet
    if (m_commandBuffers.empty())
    {
        init();
    }

    auto& inFlightFence = m_inFlightFences[m_imageIndex];
    auto& presentComplete = m_presentCompleteSemaphores[m_imageIndex];
    auto& renderFinished = m_renderFinishedSemaphores[m_imageIndex];
    auto& commandBuffer = m_commandBuffers[m_imageIndex];

    // Wait for fence for this frame to be signaled (previous GPU work finished)
    m_rhi.getDevice().waitForFences(*inFlightFence, true, UINT64_MAX);

    m_imageIndex = m_rhi.acquireNextSwapChainImage(*presentComplete).second;

    executeRenderPasses();

    // reset the fence to unsignaled before submit
    m_rhi.getDevice().resetFences(*inFlightFence);

    // Submit: wait on presentComplete, signal renderFinished
    vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &*presentComplete;
    submitInfo.pWaitDstStageMask = &waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*m_commandBuffers[m_imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &*renderFinished;

    m_rhi.presentSwapChainImage(m_imageIndex, submitInfo, *inFlightFence);
}

// ---------------------------------------------------------------------------
// Reender pass execution with automatic image layout transitions
// ---------------------------------------------------------------------------

void RenderGraph::executeRenderPasses()
{
    // ---- Per-frame image layout state tracking ----
    // Only images that appear as render targets in ANY pass are tracked.
    // Static textures (shader-read only, never a render target) are assumed to already
    // be in eShaderReadOnlyOptimal and are not transitioned.
    std::unordered_map<VkImage, vk::ImageLayout> imageLayouts{};
    auto swapChainColorLayout = vk::ImageLayout::eUndefined;
    auto swapChainDepthLayout = vk::ImageLayout::eUndefined;

    // Which render targets have been cleared this frame (determines loadOp)
    std::unordered_set<VkImage> clearedImages{};
    auto swapChainColorCleared = false;
    auto swapChainDepthCleared = false;

    // Seed the tracking map with all render-target images across every pass
    for (const auto& renderPass : m_renderPasses)
    {
        if (std::holds_alternative<GraphicsPass>(renderPass))
        {
            const auto& graphicsPass = std::get<GraphicsPass>(renderPass);

            for (const auto& image : graphicsPass.colorTargetImages)
            {
                imageLayouts.try_emplace(image.images[0], vk::ImageLayout::eUndefined);
            }

            if (graphicsPass.depthTargetImage.images.size())
            {
                imageLayouts.try_emplace(graphicsPass.depthTargetImage.images[0], vk::ImageLayout::eUndefined);
            }
        }
    }

    auto swapChainExtent = m_rhi.getSwapChainExtent();
    auto anyPassUsedSwapChainColor = false;

    // ---- Helper: populate a barrier based on old/new layout ----
    auto addImageBarrier = [](
        std::vector<vk::ImageMemoryBarrier2>& barriers,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::ImageAspectFlags aspectFlags)
    {
        vk::ImageMemoryBarrier2 barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspectFlags;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        // Source access / stage (what produced the previous content)
        switch (oldLayout)
        {
        case vk::ImageLayout::eUndefined:
            barrier.srcAccessMask = {};
            barrier.srcStageMask = (aspectFlags & vk::ImageAspectFlagBits::eDepth)
                ? (vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests)
                : vk::PipelineStageFlagBits2::eTopOfPipe;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            break;
        case vk::ImageLayout::eDepthAttachmentOptimal:
            barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            break;
        default:
            break;
        }

        // Destination access / stage (what will consume the image next)
        switch (newLayout)
        {
        case vk::ImageLayout::eColorAttachmentOptimal:
            barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            if (oldLayout == newLayout) // WAW hazard - also need read for blending
            {
                barrier.dstAccessMask |= vk::AccessFlagBits2::eColorAttachmentRead;
            }
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            break;
        case vk::ImageLayout::eDepthAttachmentOptimal:
            barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            if (oldLayout == newLayout)
            {
                barrier.dstAccessMask |= vk::AccessFlagBits2::eDepthStencilAttachmentRead;
            }
            barrier.dstStageMask = 
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | 
                vk::PipelineStageFlagBits2::eLateFragmentTests;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            break;
        case vk::ImageLayout::ePresentSrcKHR:
            barrier.dstAccessMask = {};
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
            break;
        default:
            break;
        }

        barriers.push_back(barrier);
    };

    const auto& commandBuffer = m_commandBuffers[m_imageIndex];

    commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // ================================================================
    // Per-pass loop
    // ================================================================
    for (const auto& renderPass : m_renderPasses)
    {
        if (std::holds_alternative<ComputePass>(renderPass))
        {
            const auto& computePass = std::get<ComputePass>(renderPass);

            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePass.pipeline);

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                computePass.pipeline.getPipelineLayout(),
                0,
                *computePass.descriptorSets[m_imageIndex],
                nullptr);

            // shader uses [numthreads(64,1,1)], so ceil(instanceCount / 64) groups in X
            commandBuffer.dispatch((computePass.minDispatchThreadCount + 63) / 64, 1, 1);

            continue;
        }

        const auto& graphicsPass = std::get<GraphicsPass>(renderPass);

        std::vector<vk::ImageMemoryBarrier2> barriers{};

        // ---- 1. Shader-read transitions (render targets from earlier passes) ----
        for (const auto& image : graphicsPass.shaderReadImages)
        {
            auto it = imageLayouts.find(image.images[0]);
            if (it != imageLayouts.end() && it->second != vk::ImageLayout::eShaderReadOnlyOptimal)
            {
                bool isDepth = !!(image.createInfo.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment);
                auto aspectFlags = isDepth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
                addImageBarrier(
                    barriers, 
                    image.images[m_imageIndex],
                    it->second, 
                    vk::ImageLayout::eShaderReadOnlyOptimal, 
                    aspectFlags);
                it->second = vk::ImageLayout::eShaderReadOnlyOptimal;
            }
            // Images NOT in the tracking map are static textures - no barrier needed.
        }

        // ---- 2. Color render-target transitions ----
        for (const auto& image : graphicsPass.colorTargetImages)
        {
            auto& layout = imageLayouts[image.images[0]];
            addImageBarrier(
                barriers, 
                image.images[m_imageIndex],
                layout, vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageAspectFlagBits::eColor);
            layout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // ---- 3. Depth render-target transition (user image) ----
        if (graphicsPass.depthTargetImage.images.size())
        {
            auto& layout = imageLayouts[graphicsPass.depthTargetImage.images[0]];
            addImageBarrier(
                barriers, 
                graphicsPass.depthTargetImage.images[m_imageIndex],
                layout, 
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::ImageAspectFlagBits::eDepth);
            layout = vk::ImageLayout::eDepthAttachmentOptimal;
        }

        // ---- 4. Swap-chain color transition ----
        if (graphicsPass.usesSwapChainColor)
        {
            anyPassUsedSwapChainColor = true;
            addImageBarrier(
                barriers, 
                m_rhi.getSwapChainImage(m_imageIndex),
                swapChainColorLayout, 
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageAspectFlagBits::eColor);
            swapChainColorLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // ---- 5. Swap-chain depth transition ----
        if (graphicsPass.usesSwapChainDepth)
        {
            addImageBarrier(
                barriers, 
                m_rhi.getDepthImage(m_imageIndex),
                swapChainDepthLayout, 
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::ImageAspectFlagBits::eDepth);
            swapChainDepthLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        }

        // ---- Issue barriers ----
        if (!barriers.empty())
        {
            vk::DependencyInfo depInfo{};
            depInfo.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            depInfo.pImageMemoryBarriers = barriers.data();
            commandBuffer.pipelineBarrier2(depInfo);
        }

        // ---- Viewport & scissor ----
        commandBuffer.setViewport(
            0,
            vk::Viewport{
                0.0f,
                0.0f,
                static_cast<float>(swapChainExtent.width),
                static_cast<float>(swapChainExtent.height),
                0.0f,
                1.0f,
            });
        commandBuffer.setScissor(0, vk::Rect2D{ { 0, 0 }, swapChainExtent });

        // ---- Build rendering attachment infos ----
        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        std::vector<vk::RenderingAttachmentInfo> colorAttachmentInfos;

        for (const auto& image : graphicsPass.colorTargetImages)
        {
            bool firstUse = clearedImages.insert(image.images[m_imageIndex]).second;
            vk::RenderingAttachmentInfo info{};
            info.imageView = image.imageViews[m_imageIndex];
            info.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
            info.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
            info.storeOp = vk::AttachmentStoreOp::eStore;
            info.clearValue = clearColor;
            colorAttachmentInfos.push_back(info);
        }

        if (graphicsPass.usesSwapChainColor)
        {
            bool firstUse = !swapChainColorCleared;
            swapChainColorCleared = true;
            vk::RenderingAttachmentInfo info{};
            info.imageView = m_rhi.getSwapChainImageView(m_imageIndex);
            info.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
            info.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
            info.storeOp = vk::AttachmentStoreOp::eStore;
            info.clearValue = clearColor;
            colorAttachmentInfos.push_back(info);
        }

        // Depth attachment (user image OR swap-chain depth, at most one)
        vk::RenderingAttachmentInfo depthAttachmentInfo{};
        bool hasDepth = (graphicsPass.depthTargetImage.images.size()) || graphicsPass.usesSwapChainDepth;

        if (hasDepth)
        {
            vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

            if (graphicsPass.depthTargetImage.images.size())
            {
                bool firstUse = clearedImages.insert(graphicsPass.depthTargetImage.images[m_imageIndex]).second;
                depthAttachmentInfo.imageView = graphicsPass.depthTargetImage.imageViews[m_imageIndex];
                depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
                depthAttachmentInfo.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
                depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
                depthAttachmentInfo.clearValue = clearDepth;
            }
            else // swap-chain depth
            {
                bool firstUse = !swapChainDepthCleared;
                swapChainDepthCleared = true;
                depthAttachmentInfo.imageView = m_rhi.getDepthImageView(m_imageIndex);
                depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
                depthAttachmentInfo.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
                depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
                depthAttachmentInfo.clearValue = clearDepth;
            }
        }

        // ---- Begin dynamic rendering ----
        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea.extent = swapChainExtent;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachmentInfos.size());
        renderingInfo.pColorAttachments = colorAttachmentInfos.data();
        if (hasDepth)
        {
            renderingInfo.pDepthAttachment = &depthAttachmentInfo;
        }

        commandBuffer.beginRendering(renderingInfo);

        // ---- Bind pipeline ----
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPass.pipeline);

        // ---- Bind descriptor sets ----
        if (!graphicsPass.descriptorSets.empty())
        {
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                graphicsPass.pipeline.getPipelineLayout(),
                0,
                *graphicsPass.descriptorSets[m_imageIndex],
                nullptr);
        }

        // ---- Draw ----
        if (graphicsPass.drawCommandBuffer.buffers.size())
        {
            // Indexed indirect draw (mesh geometry)
            if (graphicsPass.vertexBuffer.buffers.size())
            {
                commandBuffer.bindVertexBuffers(0, graphicsPass.vertexBuffer.buffers[0], {vk::DeviceSize(0)});
            }

            if (graphicsPass.indexBuffer.buffers.size())
            {
                commandBuffer.bindIndexBuffer(graphicsPass.indexBuffer.buffers[0], 0, vk::IndexType::eUint32);
            }

            auto drawCount = static_cast<uint32_t>(graphicsPass.drawCommandBuffer.createInfo.size / sizeof(vk::DrawIndexedIndirectCommand));
            commandBuffer.drawIndexedIndirect(
                graphicsPass.drawCommandBuffer.buffers[0],
                0,
                drawCount,
                static_cast<uint32_t>(sizeof(vk::DrawIndexedIndirectCommand)));
        }
        else
        {
            // Full-screen triangle (no vertex data needed)
            commandBuffer.draw(3, 1, 0, 0);
        }

        // ---- End rendering ----
        commandBuffer.endRendering();
    }

    // ---- Final transition: swap-chain color -> present ----
    if (anyPassUsedSwapChainColor)
    {
        vk::ImageMemoryBarrier2 presentBarrier{};
        presentBarrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        presentBarrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        presentBarrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        presentBarrier.dstAccessMask = {};
        presentBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        presentBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        presentBarrier.image = m_rhi.getSwapChainImage(m_imageIndex);
        presentBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        presentBarrier.subresourceRange.levelCount = 1;
        presentBarrier.subresourceRange.layerCount = 1;

        vk::DependencyInfo depInfo{};
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &presentBarrier;
        commandBuffer.pipelineBarrier2(depInfo);
    }

    commandBuffer.end();
}
