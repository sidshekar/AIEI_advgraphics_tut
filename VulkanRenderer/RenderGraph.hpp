#pragma once

#include <functional>

#include "RHI.hpp"
#include "Pipeline.hpp"
#include "DescriptorSet.hpp"
#include "Image.hpp"
#include "Buffer.hpp"

namespace Gfx
{
    class RenderGraph;

    template<typename PipelineCreateInfo>
    class PipelineBuilder
    {
    protected:
        PipelineBuilder(const std::string& name) : m_name(name) {}

        PipelineCreateInfo m_pipelineCreateInfo;
        std::vector<DescriptorBinding> m_descriptorBindings;

        std::string m_name;
    };

    class ComputePipelineBuilder : public PipelineBuilder<ComputePipelineCreateInfo>
    {
    private:
        friend class RenderGraph;

        ComputePipelineBuilder(const std::string& name, uint32_t minDispatchThreadCount) :
            PipelineBuilder(name),
            m_minDispatchThreadCount(minDispatchThreadCount)
        {}

    public:
        ComputePipelineBuilder& shader(std::string name)
        {
            m_pipelineCreateInfo.shader = name;
            return *this;
        }

        ComputePipelineBuilder& shaderBinding(const Buffer& buffer);

    private:
        uint32_t m_minDispatchThreadCount;
    };

    class GraphicsPipelineBuilder : public PipelineBuilder<GraphicsPipelineCreateInfo>
    {
    private:
        friend class RenderGraph;

        GraphicsPipelineBuilder(const std::string& name, vk::Format swapChainColorFormat, vk::Format swapChainDepthFormat) :
            PipelineBuilder(name),
            m_swapChainColorFormat(swapChainColorFormat),
            m_swapChainDepthFormat(swapChainDepthFormat)
        {}

    public:
        GraphicsPipelineBuilder& vertexShader(std::string name)
        {
            m_pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eVertex);
            return *this;
        }

        template<typename T>
        GraphicsPipelineBuilder& vertexBuffer(const Buffer& buffer)
        {
            m_pipelineCreateInfo.vertexInputBinding = T::getBindingDescription();
            m_pipelineCreateInfo.vertexInputAttributes = T::getAttributeDescriptions();
            m_vertexBuffer = std::move(buffer.getInfo());
            return *this;
        }

        GraphicsPipelineBuilder& indexBuffer(const Buffer& buffer)
        {
            m_indexBuffer = std::move(buffer.getInfo());
            return *this;
        }

        GraphicsPipelineBuilder& drawCommandBuffer(const Buffer& buffer)
        {
            m_drawCommandBuffer = std::move(buffer.getInfo());
            return *this;
        }

        GraphicsPipelineBuilder& fragmentShader(std::string name)
        {
            m_pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eFragment);
            return *this;
        }

        GraphicsPipelineBuilder& vertexShaderBinding(const Buffer& buffer)
        {
            return shaderBinding(&buffer, vk::ShaderStageFlagBits::eVertex);
        }

        GraphicsPipelineBuilder& fragmentShaderBinding(const Buffer& buffer)
        {
            return shaderBinding(&buffer, vk::ShaderStageFlagBits::eFragment);
        }

        GraphicsPipelineBuilder& fragmentShaderBinding(const Image& image, const Sampler& sampler = nullptr)
        {
            return shaderBinding(&image, vk::ShaderStageFlagBits::eFragment, sampler);
        }

        GraphicsPipelineBuilder& fragmentShaderBinding(const std::vector<Image>& images, const Sampler& sampler = nullptr)
        {
            return shaderBinding(&images, vk::ShaderStageFlagBits::eFragment, sampler);
        }

        GraphicsPipelineBuilder& allShadersBinding(const Buffer& buffer)
        {
            return shaderBinding(&buffer, vk::ShaderStageFlagBits::eAllGraphics);
        }

        GraphicsPipelineBuilder& renderTarget(const Image& image)
        {
            const auto& createInfo = image.getCreateInfo();

            if (createInfo.usage & vk::ImageUsageFlagBits::eColorAttachment)
            {
                m_pipelineCreateInfo.colorAttachments.emplace_back(createInfo.format);
                m_colorTargetImages.emplace_back(std::move(image.getInfo()));
            }
            else
            {
                m_pipelineCreateInfo.depthAttachment = createInfo.format;
                m_depthTargetImage = std::move(image.getInfo());
            }
            return *this;
        }

        GraphicsPipelineBuilder& renderTargetSwapChainColor()
        {
            m_pipelineCreateInfo.colorAttachments.emplace_back(m_swapChainColorFormat);
            m_usesSwapChainColor = true;
            return *this;
        }

        GraphicsPipelineBuilder& renderTargetSwapChainDepth()
        {
            m_pipelineCreateInfo.depthAttachment = m_swapChainDepthFormat;
            m_usesSwapChainDepth = true;
            return *this;
        }

    private:
        GraphicsPipelineBuilder& shaderBinding(std::variant<const std::vector<Image>*, const Image*> images, vk::ShaderStageFlagBits stage, const Sampler& sampler = nullptr);

        GraphicsPipelineBuilder& shaderBinding(const Buffer* buffer, vk::ShaderStageFlagBits stage);

    private:
        vk::Format m_swapChainColorFormat;
        vk::Format m_swapChainDepthFormat;

        BufferInfo m_vertexBuffer{};
        BufferInfo m_indexBuffer{};
        BufferInfo m_drawCommandBuffer{};

        std::vector<ImageInfo> m_colorTargetImages{};
        ImageInfo m_depthTargetImage{};
        bool m_usesSwapChainColor = false;
        bool m_usesSwapChainDepth = false;

        std::vector<ImageInfo> m_shaderReadImages{};
    };

    // ---- Render pass data stored by the render graph ----

    struct ComputePass
    {
        std::string name;
        uint32_t minDispatchThreadCount;
        Pipeline pipeline;
        std::vector<DescriptorSet> descriptorSets;
    };

    struct GraphicsPass
    {
        std::string name;
        Pipeline pipeline;
        std::vector<DescriptorSet> descriptorSets;
        BufferInfo vertexBuffer;
        BufferInfo indexBuffer;
        BufferInfo drawCommandBuffer;
        std::vector<ImageInfo> colorTargetImages;
        ImageInfo depthTargetImage;
        bool usesSwapChainColor;
        bool usesSwapChainDepth;
        std::vector<ImageInfo> shaderReadImages;
    };

    class RenderGraph
    {
    public:
        RenderGraph(RHI& rhi) : m_rhi(rhi) {}
        RenderGraph(const RenderGraph&) = delete;

        ComputePipelineBuilder& computePass(const std::string& name, uint32_t minDispatchThreadCount)
        {
            m_pipelineBuilders.emplace_back(ComputePipelineBuilder(name, minDispatchThreadCount));
            return std::get<ComputePipelineBuilder>(m_pipelineBuilders.back());
        }

        GraphicsPipelineBuilder& graphicsPass(const std::string& name)
        {
            m_pipelineBuilders.emplace_back(GraphicsPipelineBuilder(name, m_rhi.getSurfaceFormat(), m_rhi.getDepthFormat()));
            return std::get<GraphicsPipelineBuilder>(m_pipelineBuilders.back());
        }

        // Initialize per-frame resources (command buffers, semaphores, fences).
        // Must be called after creating swapchain and image views.
        void init();

        // Execute full frame: acquire, record each pass, submit, present.
        // When compiled passes are registered (via graphicsPass().build()), they are executed
        // with automatic image layout transitions. Otherwise, legacy RenderPassNode passes are used.
        void executeFrame();

        uint64_t getFrameIndex() const { return m_imageIndex; }

    private:
        void executeRenderPasses();

    private:
        RHI& m_rhi;

        std::vector<std::variant<ComputePass, GraphicsPass>> m_renderPasses;

        std::vector<std::variant<ComputePipelineBuilder, GraphicsPipelineBuilder>> m_pipelineBuilders;

        std::vector<vk::raii::CommandBuffer> m_commandBuffers;

        // per-frame synchronization objects
        std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::raii::Fence> m_inFlightFences;

        int m_imageIndex = 0;
    };
}
