#pragma once

#include <optional>
#include <variant>
#include <vulkan/vulkan_raii.hpp>

#include "Buffer.hpp"
#include "Image.hpp"
#include "DescriptorSet.hpp"
#include "Pipeline.hpp"

namespace Gfx
{
	class Buffer;
	class Image;
	class DescriptorSet;
	class Pipeline;

	struct GraphicsPipelineCreateInfo
	{
		std::vector<std::pair<std::string, vk::ShaderStageFlagBits>> shaders;
		vk::VertexInputBindingDescription vertexInputBinding;
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes;
		std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
		std::vector<vk::Format> colorAttachments;
		std::optional<vk::Format> depthAttachment;
	};

	struct ComputePipelineCreateInfo
	{
		std::string shader;
		std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
	};

	struct DescriptorBinding
	{
		vk::DescriptorType type;
		std::variant<
			std::vector<vk::DescriptorBufferInfo>,
			std::vector<std::vector<vk::DescriptorImageInfo>>
		> data;
	};

	class RHI
	{
	public:
		RHI() = default;
		RHI(const RHI&) = delete;

		void init(const std::string& appName, const std::vector<const char*>& extensions, void* window);

		const vk::raii::PhysicalDevice& getPhysicalDevice() const { return m_physicalDevice; }
		const vk::raii::Device& getDevice() const { return m_device; }
		uint8_t getMaxFramesInFlight() const { return m_maxFramesInFlight; }
		const vk::raii::CommandPool& getCommandPool() const { return m_commandPool; }
		vk::Format getSurfaceFormat() const { return m_surfaceFormat.format; }
		vk::Format getDepthFormat() const { return m_depthFormat; }
		vk::Image getSwapChainImage(int index) const { return m_swapChainImages[index]; }
		const vk::Image& getDepthImage(int index) const { return m_depthImage.getImage(index); }
		const vk::ImageView& getSwapChainImageView(int index) const { return *m_swapChainImageViews[index]; }
		const vk::ImageView& getDepthImageView(int index) const { return m_depthImage.getImageView(index); }
		vk::Extent2D getSwapChainExtent() const { return m_swapChainExtent; }

		std::pair<vk::Result, uint32_t> acquireNextSwapChainImage(const vk::Semaphore& signal) const { return m_swapChain.acquireNextImage(UINT64_MAX, signal, nullptr); }

		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, const void* contentData, size_t contentSize, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		void updateBuffer(const Buffer& buffer, const void* contentData, size_t contentSize);

		Image createImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		Sampler createSampler(const vk::SamplerCreateInfo& samplerInfo);
		void updateImage(const Gfx::Image& image, const void* contentData, size_t contentSize);

		Pipeline createPipeline(const GraphicsPipelineCreateInfo& createInfo);
		Pipeline createPipeline(const ComputePipelineCreateInfo& createInfo);

		std::vector<DescriptorSet> createDescriptorSets(const vk::DescriptorSetLayout& layout, const std::vector<DescriptorBinding>& bindings);

		template<typename T>
		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, const T& data, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal) {
		    return createBuffer(bufferInfo, &data, sizeof(T), memProperties);
		}

		template<typename T>
		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, const std::vector<T>& data, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal) {
		    return createBuffer(bufferInfo, data.data(), data.size() * sizeof(T), memProperties);
		}

		Image createImage2D(vk::Format format, vk::Extent2D extent, vk::ImageUsageFlags usageFlags)
		{
			vk::ImageCreateInfo imageInfo{};
			imageInfo.imageType = vk::ImageType::e2D;
			imageInfo.format = format;
			imageInfo.extent = vk::Extent3D{ extent, 1 };
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.samples = vk::SampleCountFlagBits::e1;
			imageInfo.usage = usageFlags;
			return createImage(imageInfo);
		}

		Image createImage2D(vk::Format format, vk::ImageUsageFlags usageFlags)
		{
			return createImage2D(format, m_swapChainExtent, usageFlags);
		}

		template<typename T>
		void updateBuffer(const Buffer& buffer, const T& data) {
			updateBuffer(buffer, &data, sizeof(T));
		}

		template<typename T>
		void updateBuffer(const Buffer& buffer, const std::vector<T>& data) {
			updateBuffer(buffer, data.data(), data.size() * sizeof(T));
		}

		void updateImage(const Image& image, const std::vector<uint8_t>& data) {
			updateImage(image, data.data(), data.size());
		}

		void presentSwapChainImage(uint32_t imageIndex, const vk::SubmitInfo& submitInfo, const vk::Fence& inFlightFence) const;

	private:
		void initInstance(const std::string& appName, const std::vector<const char*>& extensions);
		void initSurface(void* window);
		void pickPhysicalDevice();
		void initLogicalDevice();
		void initSwapChain(void* window);
		void initDepthResources();
		void initCommandPool();

	private:
		vk::raii::Context m_context{};
		vk::raii::Instance m_instance = nullptr;
		vk::raii::SurfaceKHR m_surface = nullptr;
		vk::raii::PhysicalDevice m_physicalDevice = nullptr;
		vk::raii::Device m_device = nullptr;
		uint32_t m_graphicsFamily = 0;
		uint32_t m_presentFamily = 0;
		vk::raii::Queue m_graphicsQueue = nullptr;
		vk::raii::Queue m_presentQueue = nullptr;
		vk::SurfaceFormatKHR m_surfaceFormat{};
		vk::Extent2D m_swapChainExtent{};
		vk::raii::SwapchainKHR m_swapChain = nullptr;
		uint8_t m_maxFramesInFlight = 0;
		std::vector<vk::Image> m_swapChainImages{};
		std::vector<vk::raii::ImageView> m_swapChainImageViews{};
		vk::Format m_depthFormat{};
		Gfx::Image m_depthImage = nullptr;
		vk::raii::CommandPool m_commandPool = nullptr;
	};
}
