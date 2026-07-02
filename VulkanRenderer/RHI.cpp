#include "RHI.hpp"

#include <fstream>
#include <memory>
#include <unordered_map>
#include <windows.h>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include <glm/glm.hpp>

//#include "Buffer.hpp"
//#include "Image.hpp"


using Gfx::RHI;

#undef max

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
    //vk::KHRSpirv14ExtensionName,
    //vk::KHRSynchronization2ExtensionName,
    //vk::KHRCreateRenderpass2ExtensionName,
};

static std::tuple<uint32_t, uint32_t> findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::SurfaceKHR& surface) {
    // find the index of the first queue family that supports graphics
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports graphics
    auto graphicsQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
        [](vk::QueueFamilyProperties& qfp)
        {
            return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
        });

    auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));

    // determine a queueFamilyIndex that supports present
    // first check if the graphicsIndex is good enough
    auto presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsIndex, *surface)
        ? graphicsIndex
        : static_cast<uint32_t>(queueFamilyProperties.size());

    if (presentIndex == queueFamilyProperties.size())
    {
        // the graphicsIndex doesn't support present -> look for another family index that supports both
        // graphics and present
        for (size_t i = 0; i < queueFamilyProperties.size(); i++)
        {
            if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
            {
                graphicsIndex = static_cast<uint32_t>(i);
                presentIndex = graphicsIndex;
                break;
            }
        }
        if (presentIndex == queueFamilyProperties.size())
        {
            // there's nothing like a single family index that supports both graphics and present -> look for another
            // family index that supports present
            for (size_t i = 0; i < queueFamilyProperties.size(); i++)
            {
                if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
                {
                    presentIndex = static_cast<uint32_t>(i);
                    break;
                }
            }
        }
    }

    if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size()))
    {
        throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
    }

    return { graphicsIndex, presentIndex };
}

static vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

static vk::Extent2D chooseSwapExtent(void* window, const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    RECT rect;
    GetClientRect(static_cast<HWND>(window), &rect);
    uint32_t width = rect.right - rect.left;
    uint32_t height = rect.bottom - rect.top;

    return {
        glm::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        glm::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

static uint32_t chooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities) {
    auto minImageCount = surfaceCapabilities.minImageCount + 1;
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

static vk::Format findSupportedFormat(const vk::raii::PhysicalDevice& physicalDevice,
                                      const std::vector<vk::Format>& candidates,
                                      vk::ImageTiling tiling,
                                      vk::FormatFeatureFlags features)
{
    for (const auto format : candidates) {
        auto props = physicalDevice.getFormatProperties(format);
        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

static vk::Format findDepthFormat(const vk::raii::PhysicalDevice& physicalDevice) {
    return findSupportedFormat(
        physicalDevice,
        { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

static uint32_t findMemoryType(const vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

static void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer, const vk::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.image = image;
    barrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    std::vector<char> buffer(file.tellg());

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();

    return buffer;
}

static vk::raii::ShaderModule createShaderModule(const vk::raii::Device& device, const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size() * sizeof(char);
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    return vk::raii::ShaderModule{ device, createInfo };
}

void RHI::init(const std::string& appName, const std::vector<const char*>& extensions, void* window) {
    initInstance(appName, extensions);
    initSurface(window);
    pickPhysicalDevice();
    initLogicalDevice();
    initSwapChain(window);
    initDepthResources();
    initCommandPool();
}

void RHI::initInstance(const std::string& appName, const std::vector<const char*>& extensions) {
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Vulkan Renderer";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = vk::ApiVersion13;

    // Get the required layers
    std::vector<char const*> requiredLayers{};
    if (enableValidationLayers) {
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = m_context.enumerateInstanceLayerProperties();
    if (std::any_of(requiredLayers.begin(), requiredLayers.end(),
        [&layerProperties](auto const& requiredLayer)
        {
            return std::none_of(layerProperties.begin(), layerProperties.end(),
                [requiredLayer](auto const& layerProperty)
                {
                    return strcmp(layerProperty.layerName, requiredLayer) == 0;
                });
        }))
    {
        throw std::runtime_error("One or more required layers are not supported!");
    }

    // Check if the required GLFW extensions are supported by the Vulkan implementation.
    auto extensionProperties = m_context.enumerateInstanceExtensionProperties();
    for (size_t i = 0; i < extensions.size(); ++i)
    {
        auto extension = extensions[i];
        if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
            [extension](auto const& extensionProperty)
            {
                return std::strcmp(extensionProperty.extensionName, extension) == 0;
            }))
        {
            throw std::runtime_error(std::string("Required GLFW extension not supported: ") + extension);
        }
    }

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
    createInfo.ppEnabledLayerNames = requiredLayers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    m_instance = vk::raii::Instance(m_context, createInfo);
}

void RHI::initSurface(void* window)
{
    vk::Win32SurfaceCreateInfoKHR createInfo{};
    createInfo.sType = vk::StructureType::eWin32SurfaceCreateInfoKHR;
    createInfo.hwnd = static_cast<HWND>(window);
    createInfo.hinstance = GetModuleHandle(nullptr);

    m_surface = vk::raii::SurfaceKHR(m_instance, createInfo);
}

void RHI::pickPhysicalDevice() {
    auto devices = m_instance.enumeratePhysicalDevices();
    auto devIter = std::find_if(devices.begin(), devices.end(),
        [&](auto& device)
        {
            auto queueFamilies = device.getQueueFamilyProperties();
            auto isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;
            auto qfpIter = std::find_if(queueFamilies.begin(), queueFamilies.end(),
                [](vk::QueueFamilyProperties& qfp)
                {
                    return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                });
            isSuitable = isSuitable && (qfpIter != queueFamilies.end());
            auto extensions = device.enumerateDeviceExtensionProperties();
            auto found = true;
            for (auto& extension : deviceExtensions) {
                auto extensionIter = std::find_if(extensions.begin(), extensions.end(),
                    [extension](auto& ext)
                    {
                        return strcmp(ext.extensionName, extension) == 0;
                    });
                found = found && extensionIter != extensions.end();
            }
            isSuitable = isSuitable && found;
            if (isSuitable) {
                m_physicalDevice = device;
            }
            return isSuitable;
        });
    if (devIter == devices.end()) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void RHI::initLogicalDevice() {
    std::tie(m_graphicsFamily, m_presentFamily) = findQueueFamilies(m_physicalDevice, m_surface);
    auto queuePriority = 0.0f;

    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
    deviceQueueCreateInfo.queueFamilyIndex = m_graphicsFamily;
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

    vk::PhysicalDeviceFeatures2 features2{};
    features2.features.samplerAnisotropy = true;
    features2.features.multiDrawIndirect = true;

    vk::PhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.runtimeDescriptorArray = true;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing = true;

    vk::PhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.dynamicRendering = true; // Enable dynamic rendering from Vulkan 1.3
    vulkan13Features.synchronization2 = true; // enable synchronization2 from the extension

    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extDynamicStateFeatures{};
    extDynamicStateFeatures.extendedDynamicState = true; // Enable extended dynamic state from the extension

    // Create a chain of feature structures
    auto featureChain = vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
    { features2, vulkan12Features, vulkan13Features, extDynamicStateFeatures };

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    m_device = vk::raii::Device(m_physicalDevice, deviceCreateInfo);
    m_graphicsQueue = vk::raii::Queue(m_device, m_graphicsFamily, 0);
    m_presentQueue = vk::raii::Queue(m_device, m_presentFamily, 0);
}

void RHI::initSwapChain(void* window) {
    auto surfaceCapabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(m_surface);
    auto availableFormats = m_physicalDevice.getSurfaceFormatsKHR(m_surface);
    auto availablePresentModes = m_physicalDevice.getSurfacePresentModesKHR(m_surface);

    m_surfaceFormat = chooseSurfaceFormat(availableFormats);
    m_swapChainExtent = chooseSwapExtent(window, surfaceCapabilities);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.flags = vk::SwapchainCreateFlagsKHR();
    swapChainCreateInfo.surface = m_surface;
    swapChainCreateInfo.minImageCount = chooseSwapMinImageCount(surfaceCapabilities);
    swapChainCreateInfo.imageFormat = m_surfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = m_surfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent = m_swapChainExtent;
    swapChainCreateInfo.imageArrayLayers = 1; // keep 1 unless rendering for VR
    swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // we are rendering to image directly
    swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;  // don't apply further transformation
    swapChainCreateInfo.presentMode = chooseSwapPresentMode(availablePresentModes);
    swapChainCreateInfo.clipped = true;  // don't update the pixels that are obscured

    uint32_t queueFamilyIndices[] = { m_graphicsFamily, m_presentFamily };

    if (m_graphicsFamily != m_presentFamily) {
        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        swapChainCreateInfo.queueFamilyIndexCount = 2;
        swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
    }

    m_swapChain = vk::raii::SwapchainKHR(m_device, swapChainCreateInfo);

    m_swapChainImages = m_swapChain.getImages();

    m_maxFramesInFlight = static_cast<uint8_t>(m_swapChainImages.size());

    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imageViewCreateInfo.format = m_surfaceFormat.format;
    imageViewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    for (auto image : m_swapChainImages) {
        imageViewCreateInfo.image = image;
        m_swapChainImageViews.emplace_back(m_device, imageViewCreateInfo);
    }
}

void RHI::initDepthResources()
{
    m_depthFormat = findDepthFormat(m_physicalDevice);

    vk::ImageCreateInfo depthImageInfo{};
    depthImageInfo.imageType = vk::ImageType::e2D;
    depthImageInfo.format = m_depthFormat;
    depthImageInfo.extent.width = m_swapChainExtent.width;
    depthImageInfo.extent.height = m_swapChainExtent.height;
    depthImageInfo.extent.depth = 1;
    depthImageInfo.mipLevels = 1;
    depthImageInfo.arrayLayers = 1;
    depthImageInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

    m_depthImage = createImage(depthImageInfo);
}

void RHI::initCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = m_graphicsFamily;

    m_commandPool = vk::raii::CommandPool(m_device, poolInfo);
}

Gfx::Buffer RHI::createBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties)
{
    auto bufferedCount =
        (bufferInfo.usage & vk::BufferUsageFlagBits::eUniformBuffer) ?
        m_maxFramesInFlight :
        1;

    std::vector<vk::raii::Buffer> buffers;
    std::vector<vk::raii::DeviceMemory> bufferMemories;

    buffers.reserve(bufferedCount);
    bufferMemories.reserve(bufferedCount);

    for (size_t i = 0; i < bufferedCount; i++)
    {
        vk::raii::Buffer buffer(m_device, bufferInfo);

        auto memRequirements = buffer.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, memProperties);

        vk::raii::DeviceMemory bufferMemory(m_device, allocInfo);

        buffer.bindMemory(bufferMemory, 0);

        buffers.emplace_back(std::move(buffer));
        bufferMemories.emplace_back(std::move(bufferMemory));
    }

    return Gfx::Buffer(bufferInfo, std::move(buffers), std::move(bufferMemories));
}

Gfx::Buffer RHI::createBuffer(const vk::BufferCreateInfo& bufferInfo, const void* contentData, size_t contentSize, vk::MemoryPropertyFlags memProperties)
{
    auto createInfo = bufferInfo;
    if (!(createInfo.usage & vk::BufferUsageFlagBits::eTransferDst))
    {
        createInfo.usage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    auto buffer = createBuffer(createInfo, memProperties);
    updateBuffer(buffer, contentData, contentSize);
    return buffer;
}

void RHI::updateBuffer(const Buffer& buffer, const void* contentData, size_t contentSize)
{
    vk::BufferCreateInfo stagingInfo{};
    stagingInfo.size = contentSize;
    stagingInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;

    auto stagingBuffer = createBuffer(stagingInfo,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent);

    stagingBuffer.map();
    memcpy(stagingBuffer.getMappedData(0), contentData, stagingInfo.size);
    stagingBuffer.unmap();

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    auto commandCopyBuffer = std::move(m_device.allocateCommandBuffers(allocInfo).front());
    commandCopyBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    vk::BufferCopy region{};
    region.size = stagingInfo.size;
    for (int i = 0; i < buffer.getBufferCount(); i++)
    {
        commandCopyBuffer.copyBuffer(stagingBuffer.getBuffer(0), buffer.getBuffer(i), region);
    }
    commandCopyBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandCopyBuffer;

    // TODO: use a fence instead
    m_graphicsQueue.submit(submitInfo, nullptr);
    m_graphicsQueue.waitIdle();
}

Gfx::Image RHI::createImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags memProperties)
{
    auto createInfo = imageInfo;
    createInfo.usage |= vk::ImageUsageFlagBits::eSampled;

    auto bufferedCount =
        createInfo.usage & (
            vk::ImageUsageFlagBits::eColorAttachment | 
            vk::ImageUsageFlagBits::eDepthStencilAttachment) ?
        m_maxFramesInFlight :
        1;

    std::vector<vk::raii::Image> images;
    std::vector<vk::raii::DeviceMemory> imageMemories;
    std::vector<vk::raii::ImageView> imageViews;

    images.reserve(bufferedCount);
    imageMemories.reserve(bufferedCount);
    imageViews.reserve(bufferedCount);

    for (size_t i = 0; i < bufferedCount; i++)
    {
        vk::raii::Image image(m_device, createInfo);

        auto memRequirements = image.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, memProperties);

        vk::raii::DeviceMemory imageMemory(m_device, allocInfo);

        image.bindMemory(imageMemory, 0);

        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = createInfo.format;
        viewInfo.subresourceRange.aspectMask =
            (createInfo.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) ?
            vk::ImageAspectFlagBits::eDepth: 
            vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        vk::raii::ImageView imageView(m_device, viewInfo);

        images.emplace_back(std::move(image));
        imageMemories.emplace_back(std::move(imageMemory));
        imageViews.emplace_back(std::move(imageView));
    }

    return Gfx::Image(createInfo, std::move(images), std::move(imageMemories), std::move(imageViews));
}

Gfx::Sampler RHI::createSampler(const vk::SamplerCreateInfo& samplerInfo)
{
    auto info = samplerInfo;
    if (info.anisotropyEnable && info.maxAnisotropy == 0)
    {
        info.maxAnisotropy = m_physicalDevice.getProperties().limits.maxSamplerAnisotropy;
    }

    auto sampler = vk::raii::Sampler(m_device, info);

    return Gfx::Sampler(std::move(info), std::move(sampler));
}

void RHI::updateImage(const Gfx::Image& image, const void* contentData, size_t contentSize)
{
    vk::BufferCreateInfo stagingInfo{};
    stagingInfo.size = contentSize;
    stagingInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;

    auto stagingBuffer = createBuffer(stagingInfo,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent);

    stagingBuffer.map();
    memcpy(stagingBuffer.getMappedData(0), contentData, stagingInfo.size);
    stagingBuffer.unmap();

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    auto commandCopyBuffer = std::move(m_device.allocateCommandBuffers(allocInfo).front());
    commandCopyBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    vk::BufferImageCopy region{};
    region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
    region.imageExtent = image.m_createInfo.extent;
    for (int i = 0; i < image.getImageCount(); i++)
    {
        transitionImageLayout(commandCopyBuffer, image.getImage(i), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        commandCopyBuffer.copyBufferToImage(stagingBuffer.getBuffer(0), image.getImage(i), vk::ImageLayout::eTransferDstOptimal, { region });
        transitionImageLayout(commandCopyBuffer, image.getImage(i), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    commandCopyBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandCopyBuffer;

    // TODO: use a fence instead
    m_graphicsQueue.submit(submitInfo, nullptr);
    m_graphicsQueue.waitIdle();
}

Gfx::Pipeline RHI::createPipeline(const Gfx::GraphicsPipelineCreateInfo& createInfo)
{
    std::vector<vk::Format> colorAttachmentFormats{};
	std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments{};

	colorAttachmentFormats.reserve(createInfo.colorAttachments.size());
	colorBlendAttachments.reserve(createInfo.colorAttachments.size());

    for (const auto& colorFormat : createInfo.colorAttachments) {
        colorAttachmentFormats.emplace_back(colorFormat);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;

        colorBlendAttachments.emplace_back(std::move(colorBlendAttachment));
	}

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(createInfo.descriptorSetLayoutBindings.size());
    layoutInfo.pBindings = createInfo.descriptorSetLayoutBindings.data();

    vk::raii::DescriptorSetLayout descriptorSetLayout(m_device, layoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;

    vk::raii::PipelineLayout pipelineLayout(m_device, pipelineLayoutInfo);

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size());
    pipelineRenderingCreateInfo.pColorAttachmentFormats = colorAttachmentFormats.data();
	pipelineRenderingCreateInfo.depthAttachmentFormat =
	    createInfo.depthAttachment.has_value() ?
			*createInfo.depthAttachment :
			vk::Format::eUndefined;

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages{};
    std::vector<vk::raii::ShaderModule> shaderModules{};

	shaderStages.reserve(createInfo.shaders.size());
	shaderModules.reserve(createInfo.shaders.size());

    for (const auto& shader : createInfo.shaders) {
        auto code = readFile(shader.first);
        auto module = createShaderModule(m_device, code);

        vk::PipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.stage = shader.second;
        shaderStageInfo.module = module;
        shaderStageInfo.pName = "main";

        shaderModules.emplace_back(std::move(module));
        shaderStages.emplace_back(std::move(shaderStageInfo));
	}

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

    std::array<vk::DynamicState, 2> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState{
        {},
        static_cast<uint32_t>(dynamicStates.size()),
        dynamicStates.data()
    };

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
    colorBlending.pAttachments = colorBlendAttachments.data();

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = createInfo.depthAttachment.has_value();
    depthStencil.depthWriteEnable = depthStencil.depthTestEnable;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &createInfo.vertexInputBinding;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(createInfo.vertexInputAttributes.size());
    vertexInputInfo.pVertexAttributeDescriptions = createInfo.vertexInputAttributes.data();

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.pNext = &pipelineRenderingCreateInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = pipelineLayout;

    vk::raii::Pipeline pipeline(m_device, nullptr, pipelineInfo);

	return Gfx::Pipeline(std::move(pipeline), std::move(pipelineLayout), std::move(descriptorSetLayout));
}

Gfx::Pipeline RHI::createPipeline(const Gfx::ComputePipelineCreateInfo& createInfo)
{
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(createInfo.descriptorSetLayoutBindings.size());
    layoutInfo.pBindings = createInfo.descriptorSetLayoutBindings.data();

    vk::raii::DescriptorSetLayout descriptorSetLayout(m_device, layoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;

    vk::raii::PipelineLayout pipelineLayout(m_device, pipelineLayoutInfo);

    auto code = readFile(createInfo.shader);
    auto shaderModule = createShaderModule(m_device, code);

    vk::PipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    vk::ComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    vk::raii::Pipeline pipeline(m_device, nullptr, pipelineInfo);

    return Gfx::Pipeline(std::move(pipeline), std::move(pipelineLayout), std::move(descriptorSetLayout));
}

std::vector<Gfx::DescriptorSet> RHI::createDescriptorSets(const vk::DescriptorSetLayout& layout, const std::vector<Gfx::DescriptorBinding>& bindings)
{
    std::unordered_map<vk::DescriptorType, uint32_t> typeCounts{};

    for (const auto& binding : bindings)
    {
        uint32_t count = m_maxFramesInFlight;
        if (!std::holds_alternative<std::vector<vk::DescriptorBufferInfo>>(binding.data))
        {
            const auto& perFrameImages =
                std::get<std::vector<std::vector<vk::DescriptorImageInfo>>>(binding.data);
            auto imagesPerSet = static_cast<uint32_t>(perFrameImages[0].size());
            count *= imagesPerSet;
        }
        typeCounts[binding.type] += count;
    }

    std::vector<vk::DescriptorPoolSize> poolSizes{};
    poolSizes.reserve(typeCounts.size());

    for (const auto& [type, count] : typeCounts)
    {
        poolSizes.push_back(vk::DescriptorPoolSize{ type, count });
    }

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets       = static_cast<uint32_t>(m_maxFramesInFlight);
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();

    auto pool = std::make_shared<vk::raii::DescriptorPool>(m_device, poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(m_maxFramesInFlight, layout);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool     = *pool;
    allocInfo.descriptorSetCount = m_maxFramesInFlight;
    allocInfo.pSetLayouts        = layouts.data();

    std::vector<Gfx::DescriptorSet> descriptorSets{};
    descriptorSets.reserve(m_maxFramesInFlight);

    auto sets = m_device.allocateDescriptorSets(allocInfo);
    for (auto& set : sets)
    {
        auto descriptorSet = DescriptorSet(pool, std::move(set));

        descriptorSets.emplace_back(std::move(descriptorSet));
    }

    for (uint32_t i = 0; i < m_maxFramesInFlight; ++i)
    {
        vk::DescriptorSet dstSet = *descriptorSets[i];

        for (size_t j = 0; j < bindings.size(); ++j)
        {
            const auto& binding = bindings[j];

            vk::WriteDescriptorSet write{};
            write.dstSet = dstSet;
            write.dstBinding = static_cast<uint32_t>(j);
            write.dstArrayElement = 0;
            write.descriptorType = binding.type;

            if (std::holds_alternative<std::vector<vk::DescriptorBufferInfo>>(binding.data))
            {
                const auto& bufInfos =
                    std::get<std::vector<vk::DescriptorBufferInfo>>(binding.data);
                if (bufInfos.empty()) continue;
                // Use per-frame entry if available, otherwise fall back to index 0.
                const auto& bufInfo = (i < bufInfos.size()) ? bufInfos[i] : bufInfos[0];

                write.descriptorCount = 1;
                write.pBufferInfo = &bufInfo;
            }
            else
            {
                const auto& perFrameImages =
                    std::get<std::vector<std::vector<vk::DescriptorImageInfo>>>(binding.data);
                if (perFrameImages.empty()) { continue; }
                // Use per-frame entry if available, otherwise fall back to index 0.
                const std::vector<vk::DescriptorImageInfo>& imgInfos =
                    (i < perFrameImages.size()) ? perFrameImages[i] : perFrameImages[0];

                write.descriptorCount = static_cast<uint32_t>(imgInfos.size());
                write.pImageInfo = imgInfos.data();
            }

            m_device.updateDescriptorSets(write, {});
        }
    }

    return descriptorSets;
}

void RHI::presentSwapChainImage(uint32_t imageIndex, const vk::SubmitInfo& submitInfo, const vk::Fence& inFlightFence) const
{
    m_graphicsQueue.submit(submitInfo, inFlightFence);

    // Present: wait on renderFinished
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = submitInfo.pSignalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &*m_swapChain;
    presentInfo.pImageIndices = &imageIndex;

    m_presentQueue.presentKHR(presentInfo);
}
