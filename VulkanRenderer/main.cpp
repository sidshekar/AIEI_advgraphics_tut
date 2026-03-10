#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "RenderGraph.hpp"

#undef max

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
    //vk::KHRSpirv14ExtensionName,
    //vk::KHRSynchronization2ExtensionName,
    //vk::KHRCreateRenderpass2ExtensionName
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

struct Vertex
{
    glm::vec2 position;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, position))
        };
    }
};

const std::vector<Vertex> vertices =
{
    {{-0.5, -0.5}},
    {{0.5, -0.5}},
    {{0.5, 0.5}},
    {{-0.5, 0.5}}
};

const std::vector<uint32_t> indices =
{
    0, 1, 2, 2, 3, 0
};

const vk::DrawIndexedIndirectCommand drawCmd = {
    static_cast<uint32_t>(indices.size()), // indexCount
    1, // instanceCount
    0, // firstIndex
    0, // vertexOffset
    0  // firstInstance
};

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct StorageBufferObject
{
    glm::vec3 colour;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context context{};
    vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    uint32_t graphicsFamily = 0, presentFamily = 0;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::SurfaceFormatKHR swapChainSurfaceFormat{};
    vk::Extent2D swapChainExtent{};
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages{};
    std::vector<vk::raii::ImageView> swapChainImageViews{};
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
    vk::raii::Buffer indirectBuffer = nullptr;
    vk::raii::DeviceMemory indirectBufferMemory = nullptr;
    vk::raii::Buffer storageBuffer = nullptr;
    vk::raii::DeviceMemory storageBufferMemory = nullptr;
    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    // Removed old single command buffer and sync objects:
    // vk::raii::CommandBuffer commandBuffer = nullptr;
    // vk::raii::Semaphore presentCompleteSemaphore = nullptr;
    // vk::raii::Semaphore renderFinishedSemaphore = nullptr;
    // vk::raii::Fence drawFence = nullptr;

    // New: render graph that encapsulates per-frame sync, command buffers and simple pass graph
    std::unique_ptr<RenderGraph> renderGraph;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        // create and initialize the render graph (allocates per-image command-buffers and sync)
        initRenderGraph();
        createVertexBuffer();
        createIndexBuffer();
        createIndirectBuffer();
        createUniformBuffers();
        createStorageBuffer();
        createDescriptorPool();
        createDescriptorSets();
    }

    void createInstance() {
        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = "Hello Triangle";
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
        auto layerProperties = context.enumerateInstanceLayerProperties();
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

        // Get the required instance extensions from GLFW.
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // Check if the required GLFW extensions are supported by the Vulkan implementation.
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (uint32_t i = 0; i < glfwExtensionCount; ++i)
        {
            auto glfwExtension = glfwExtensions[i];
            if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
                [glfwExtension](auto const& extensionProperty)
                {
                    return std::strcmp(extensionProperty.extensionName, glfwExtension) == 0;
                }))
            {
                throw std::runtime_error(std::string("Required GLFW extension not supported: ") + glfwExtension);
            }
        }

        vk::InstanceCreateInfo createInfo{};
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
        createInfo.ppEnabledLayerNames = requiredLayers.data();
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        instance = vk::raii::Instance(context, createInfo);
    }

    void pickPhysicalDevice() {
        auto devices = instance.enumeratePhysicalDevices();
        const auto devIter = std::find_if(devices.begin(), devices.end(),
            [&](auto const& device)
            {
                auto queueFamilies = device.getQueueFamilyProperties();
                auto isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;
                const auto qfpIter = std::find_if(queueFamilies.begin(), queueFamilies.end(),
                    [](vk::QueueFamilyProperties const& qfp)
                    {
                        return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                    });
                isSuitable = isSuitable && (qfpIter != queueFamilies.end());
                auto extensions = device.enumerateDeviceExtensionProperties();
                auto found = true;
                for (auto const& extension : deviceExtensions) {
                    auto extensionIter = std::find_if(extensions.begin(), extensions.end(),
                        [extension](auto const& ext)
                        {
                            return strcmp(ext.extensionName, extension) == 0;
                        });
                    found = found && extensionIter != extensions.end();
                }
                isSuitable = isSuitable && found;
                if (isSuitable) {
                    physicalDevice = device;
                }
                return isSuitable;
            });
        if (devIter == devices.end()) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        std::tie(graphicsFamily, presentFamily) = findQueueFamilies(*physicalDevice);
        auto queuePriority = 0.0f;

        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
        deviceQueueCreateInfo.queueFamilyIndex = graphicsFamily;
        deviceQueueCreateInfo.queueCount = 1;
        deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

        vk::PhysicalDeviceFeatures2 features2{}; // vk::PhysicalDeviceFeatures2 (empty for now)
        vk::PhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.dynamicRendering = true; // Enable dynamic rendering from Vulkan 1.3
        vulkan13Features.synchronization2 = true; // enable synchronization2 from the extension
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extDynamicStateFeatures{};
        extDynamicStateFeatures.extendedDynamicState = true; // Enable extended dynamic state from the extension

        // Create a chain of feature structures
        auto featureChain = vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
        { features2, vulkan13Features, extDynamicStateFeatures };

        vk::DeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        graphicsQueue = vk::raii::Queue(device, graphicsFamily, 0);
        presentQueue = vk::raii::Queue(device, presentFamily, 0);
    }

    std::tuple<uint32_t, uint32_t> findQueueFamilies(VkPhysicalDevice device) {
        // find the index of the first queue family that supports graphics
        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
            [](vk::QueueFamilyProperties const& qfp)
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

    void createSurface() {
        VkWin32SurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hwnd = glfwGetWin32Window(window);
        createInfo.hinstance = GetModuleHandle(nullptr);

        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        auto availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
        auto availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);

        swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);

        vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
        swapChainCreateInfo.flags = vk::SwapchainCreateFlagsKHR();
        swapChainCreateInfo.surface = surface;
        swapChainCreateInfo.minImageCount = chooseSwapMinImageCount(surfaceCapabilities);
        swapChainCreateInfo.imageFormat = swapChainSurfaceFormat.format;
        swapChainCreateInfo.imageColorSpace = swapChainSurfaceFormat.colorSpace;
        swapChainCreateInfo.imageExtent = swapChainExtent;
        swapChainCreateInfo.imageArrayLayers = 1; // keep 1 unless rendering for VR
        swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // we are rendering to image directly
        swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;  // don't apply further transformation
        swapChainCreateInfo.presentMode = chooseSwapPresentMode(availablePresentModes);
        swapChainCreateInfo.clipped = true;  // don�t update the pixels that are obscured

        uint32_t queueFamilyIndices[] = { graphicsFamily, presentFamily };

        if (graphicsFamily != presentFamily) {
            swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            swapChainCreateInfo.queueFamilyIndexCount = 2;
            swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
        }

        swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
        auto minImageCount = surfaceCapabilities.minImageCount + 1;
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
            minImageCount = surfaceCapabilities.maxImageCount;
        }
        return minImageCount;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {
            clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
        imageViewCreateInfo.format = swapChainSurfaceFormat.format;
        imageViewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        for (auto image : swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
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

    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size() * sizeof(char);
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        return vk::raii::ShaderModule{ device, createInfo };
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding ssboLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, ssboLayoutBinding };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    void createGraphicsPipeline() {
        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
        pipelineRenderingCreateInfo.colorAttachmentCount = 1;
        pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapChainSurfaceFormat.format;

        auto fragCode = readFile("Shaders/main.frag.spv");
        auto vertCode = readFile("Shaders/main.vert.spv");

        auto fragModule = createShaderModule(fragCode);
        auto vertModule = createShaderModule(vertCode);

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragShaderStageInfo.module = fragModule;
        fragShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertShaderStageInfo.module = vertModule;
        vertShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates.size(), dynamicStates.data());

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.lineWidth = 1.0f;

        vk::PipelineMultisampleStateCreateInfo multisampling{};
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;

        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.pNext = &pipelineRenderingCreateInfo;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = graphicsFamily;

        commandPool = vk::raii::CommandPool(device, poolInfo);

        // keep old single-command allocation removed: render graph will allocate per-image command buffers
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        auto memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(const vk::BufferCreateInfo& bufferInfo, const vk::MemoryPropertyFlags& properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
        buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*bufferMemory, 0);
    }

    template<class T>
    void uploadBuffer(const std::vector<T>& contents, const vk::raii::Buffer& buffer) {
        vk::BufferCreateInfo stagingInfo{};
        stagingInfo.size = sizeof(contents[0]) * contents.size();
        stagingInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(
            stagingInfo,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, stagingInfo.size);
        memcpy(data, contents.data(), stagingInfo.size);
        stagingBufferMemory.unmapMemory();

        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;

        auto commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
        commandCopyBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        commandCopyBuffer.copyBuffer(stagingBuffer, buffer, vk::BufferCopy(0, 0, stagingInfo.size));
        commandCopyBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandCopyBuffer;

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    void createVertexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;

        createBuffer(
            bufferInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vertexBuffer,
            vertexBufferMemory);

        uploadBuffer(vertices, vertexBuffer);
    }

    void createIndexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(indices[0]) * indices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;

        createBuffer(
            bufferInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            indexBuffer,
            indexBufferMemory);

        uploadBuffer(indices, indexBuffer);
    }

    void createIndirectBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(drawCmd);
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst;

        createBuffer(
            bufferInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            indirectBuffer,
            indirectBufferMemory);

        uploadBuffer(std::vector<vk::DrawIndexedIndirectCommand>{ drawCmd }, indirectBuffer);
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(UniformBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vk::raii::Buffer uniformBuffer = nullptr;
            vk::raii::DeviceMemory uniformBufferMemory = nullptr;
            createBuffer(
                bufferInfo,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
                uniformBuffer,
                uniformBufferMemory);
            uniformBuffers.emplace_back(std::move(uniformBuffer));
            uniformBuffersMemory.emplace_back(std::move(uniformBufferMemory));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferInfo.size));
        }
    }

    void createStorageBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(StorageBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;

        createBuffer(
            bufferInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            storageBuffer,
            storageBufferMemory);

        StorageBufferObject ssboData{};
        ssboData.colour = glm::vec3(1.0f, 1.0f, 0.0f);

        uploadBuffer(std::vector<StorageBufferObject>{ssboData}, storageBuffer);
    }

    void createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(swapChainImages.size()) },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(swapChainImages.size()) }
        };

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), *descriptorSetLayout);

        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets = device.allocateDescriptorSets(allocInfo);

        // storage buffer descriptor info (same buffer for all sets)
        vk::DescriptorBufferInfo storageBufferInfo{};
        storageBufferInfo.buffer = storageBuffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = sizeof(StorageBufferObject);

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vk::DescriptorBufferInfo uboBufferInfo{};
            uboBufferInfo.buffer = uniformBuffers[i];
            uboBufferInfo.offset = 0;
            uboBufferInfo.range = sizeof(UniformBufferObject);

            vk::WriteDescriptorSet uboWrite{};
            uboWrite.dstSet = descriptorSets[i];
            uboWrite.dstBinding = 0;
            uboWrite.dstArrayElement = 0;
            uboWrite.descriptorCount = 1;
            uboWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
            uboWrite.pBufferInfo = &uboBufferInfo;

            device.updateDescriptorSets(uboWrite, {});

            vk::WriteDescriptorSet ssboWrite{};
            ssboWrite.dstSet = descriptorSets[i];
            ssboWrite.dstBinding = 1;
            ssboWrite.dstArrayElement = 0;
            ssboWrite.descriptorCount = 1;
            ssboWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
            ssboWrite.pBufferInfo = &storageBufferInfo;

            device.updateDescriptorSets(ssboWrite, {});
        }
    }

    // New: create and initialize the RenderGraph and add the passes used by the app
    void initRenderGraph()
    {
        // construct the render graph (holds references, does NOT copy objects)
        renderGraph.reset(new RenderGraph(device, swapChain, graphicsQueue, presentQueue, commandPool, swapChainImageViews, swapChainExtent));

        // Main rendering pass: transition Undefined -> ColorAttachmentOptimal and record in the pass
        RenderPassNode mainPass{};
        mainPass.name = "MainPass";
        mainPass.oldLayout = vk::ImageLayout::eUndefined;
        mainPass.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        mainPass.srcAccessMask = {}; // from undefined
        mainPass.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        mainPass.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        mainPass.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;

        // record the same rendering commands previously inside recordCommandBuffer (beginRendering, bind pipeline, draw, endRendering)
        mainPass.recordFunc = [this](vk::raii::CommandBuffer& cmd, uint32_t imageIndex)
            {
                updateUniformBuffer(imageIndex);

                vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
                vk::RenderingAttachmentInfo attachmentInfo{};
                attachmentInfo.imageView = swapChainImageViews[imageIndex];
                attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
                attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
                attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
                attachmentInfo.clearValue = clearColor;

                vk::RenderingInfo renderingInfo{};
                renderingInfo.renderArea.offset.x = 0;
                renderingInfo.renderArea.offset.y = 0;
                renderingInfo.renderArea.extent = swapChainExtent;
                renderingInfo.layerCount = 1;
                renderingInfo.colorAttachmentCount = 1;
                renderingInfo.pColorAttachments = &attachmentInfo;

                cmd.beginRendering(renderingInfo);
                cmd.bindVertexBuffers(0, *vertexBuffer, { 0 });
                cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[imageIndex], nullptr);

                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

                cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
                cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

                cmd.drawIndexedIndirect(*indirectBuffer, 0, 1, static_cast<uint32_t>(sizeof(VkDrawIndexedIndirectCommand)));

                cmd.endRendering();
            };

        renderGraph->addPass(mainPass);

        // Final transition pass: move from color attachment -> present.
        RenderPassNode presentTransition{};
        presentTransition.name = "PresentTransition";
        presentTransition.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        presentTransition.newLayout = vk::ImageLayout::ePresentSrcKHR;
        presentTransition.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        presentTransition.dstAccessMask = {};
        presentTransition.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        presentTransition.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        presentTransition.recordFunc = nullptr; // no recording, just a layout transition

        renderGraph->addPass(presentTransition);

        // finally initialize (allocates per-image command buffers and per-frame sync objects)
        renderGraph->init();
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    // removed transitionImageLayout and recordCommandBuffer methods - RenderGraph now handles transitions + per-pass recording

    // removed createSyncObjects - RenderGraph manages per-frame sync

    void drawFrame() {
        // delegate frame orchestration to the render graph
        renderGraph->executeFrame();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    }
    catch (const std::exception& e) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}