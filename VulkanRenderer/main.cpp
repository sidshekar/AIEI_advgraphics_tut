#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>



//#define VK_USE_PLATFORM_WIN32_KHR





#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32

#include <GLFW/glfw3native.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

//#include <stb_image.h>

#include "Buffer.hpp"
#include "Image.hpp"
#include "Pipeline.hpp"
#include "RenderGraph.hpp"
#include "RHI.hpp"

#undef max

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        return bindingDescription;
    }

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions() {
        return std::vector<vk::VertexInputAttributeDescription>{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)),
                vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)),
                vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
        };
    }
};

struct Texture
{
    std::vector<uint8_t> imageData;
    int width;
    int height;
};

struct Instance
{
    glm::mat4 model;
    glm::vec3 colour;
};

struct UniformBufferObject
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 lightView;
    glm::mat4 lightProj;
    glm::quat rotation;
    glm::vec4 nLightDir;
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

    Gfx::RHI rhi{};
    Gfx::RenderGraph graph{ rhi };

    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};
    std::vector<Texture> textures{};
    std::vector<vk::DrawIndexedIndirectCommand> drawCmds{};
    std::vector<Instance> instances{};

    Gfx::Pipeline mainPipeline = nullptr;
    Gfx::Pipeline shadowPipeline = nullptr;
    std::vector<Gfx::Image> textureImages{};
    vk::raii::Sampler textureSampler = nullptr;
    std::vector<Gfx::Image> shadowImages{};
    vk::raii::Sampler shadowSampler = nullptr;
    Gfx::Buffer vertexBuffer = nullptr;
    Gfx::Buffer indexBuffer = nullptr;
    Gfx::Buffer indirectBuffer = nullptr;
    Gfx::Buffer storageBuffer = nullptr;
    std::vector<Gfx::Buffer> uniformBuffers;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets{};

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Renderer", nullptr, nullptr);
    }

    void initVulkan() {
        rhi.init("Vulkan Renderer", getRequiredExtensions(), glfwGetWin32Window(window));

        loadFloor();
        loadModel();

        createGraphicsPipeline();
        createShadowPipeline();
        // create and initialize the render graph (allocates per-image command-buffers and sync)
        createTextureResources();
        createShadowResources();
        createVertexBuffer();
        createIndexBuffer();
        createIndirectBuffer();
        createUniformBuffers();
        createStorageBuffer();
        createDescriptorPool();
        createDescriptorSets();

        initRenderGraph();
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
    }

    void createGraphicsPipeline() {
        Gfx::PipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.shaders = {
            { "Shaders/main.vert.spv", vk::ShaderStageFlagBits::eVertex },
            { "Shaders/main.frag.spv", vk::ShaderStageFlagBits::eFragment },
        };
        pipelineCreateInfo.vertexInputBindings = { Vertex::getBindingDescription() };
        pipelineCreateInfo.vertexInputAttributes = Vertex::getAttributeDescriptions();
        pipelineCreateInfo.descriptorSetLayoutBindings = {
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAllGraphics, nullptr },
            { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr },
            { 2, vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(textures.size()), vk::ShaderStageFlagBits::eFragment, nullptr },
            { 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr },
        };
        pipelineCreateInfo.colorAttachments = { { rhi.getSurfaceFormat() } };
        pipelineCreateInfo.depthAttachment = { rhi.getDepthFormat() };

        mainPipeline = rhi.createGraphicsPipeline(pipelineCreateInfo);
    }

    void createShadowPipeline() {
        Gfx::PipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.shaders = {
            { "Shaders/shadow.vert.spv", vk::ShaderStageFlagBits::eVertex },
            { "Shaders/shadow.frag.spv", vk::ShaderStageFlagBits::eFragment },
        };
        pipelineCreateInfo.vertexInputBindings = { Vertex::getBindingDescription() };
        pipelineCreateInfo.vertexInputAttributes = Vertex::getAttributeDescriptions();
        pipelineCreateInfo.descriptorSetLayoutBindings = {
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAllGraphics, nullptr },
            { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr },
            { 2, vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(textures.size()), vk::ShaderStageFlagBits::eFragment, nullptr },
            { 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr },
        };
        pipelineCreateInfo.depthAttachment = { rhi.getDepthFormat() };

        shadowPipeline = rhi.createGraphicsPipeline(pipelineCreateInfo);
    }

    void loadFloor()
    {
        std::vector<Vertex> quad{
            {{-0.5, -0.5, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0}},
            {{0.5, -0.5, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0}},
            {{0.5, 0.5, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0}},
            {{-0.5, 0.5, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0}}
        };

        std::vector<uint32_t> quadIndices{
            0, 1, 2, 2, 3, 0
        };

        vk::DrawIndexedIndirectCommand drawCmd{
            static_cast<uint32_t>(quadIndices.size()), // index count
            1, // instance count
            static_cast<uint32_t>(indices.size()), // first index
            static_cast<int32_t>(vertices.size()), // vertex offset
            static_cast<uint32_t>(drawCmds.size()) // first instance
        };

        drawCmds.emplace_back(std::move(drawCmd));

        vertices.insert(vertices.end(), quad.begin(), quad.end());
        indices.insert(indices.end(), quadIndices.begin(), quadIndices.end());

        Texture texture{};

        int texChannels;
        auto pixels = stbi_load("Textures/statue.jpg", &texture.width, &texture.height, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        texture.imageData.resize(texture.width * texture.height * 4);
        memcpy(texture.imageData.data(), pixels, texture.imageData.size());

        stbi_image_free(pixels);

        textures.emplace_back(std::move(texture));

        Instance instance{};
        instance.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -0.5)) * glm::scale(glm::mat4(1.0f), glm::vec3(4.0));
        instance.colour = glm::vec3(1.0f, 1.0f, 0.0f);

        instances.emplace_back(std::move(instance));
    }

    template<typename T>
    std::vector<T> ReadAccessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
        auto& view = model.bufferViews[accessor.bufferView];
        auto& buffer = model.buffers[view.buffer];

        auto dataPtr = buffer.data.data() + view.byteOffset + accessor.byteOffset;
        auto stride = accessor.ByteStride(view);
        auto count = accessor.count;

        std::vector<T> out(count);

        if (stride == sizeof(T)) {
            // tightly packed
            memcpy(out.data(), dataPtr, count * sizeof(T));
        }
        else {
            // interleaved
            for (size_t i = 0; i < count; i++) {
                memcpy(&out[i], dataPtr + i * stride, sizeof(T));
            }
        }

        return out;
    }

    uint32_t LoadPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive)
    {
        const tinygltf::Accessor& posAcc =
            model.accessors[primitive.attributes.at("POSITION")];
        auto positions = ReadAccessor<glm::vec3>(model, posAcc);

        std::vector<glm::vec3> normals;
        if (primitive.attributes.count("NORMAL")) {
            auto& normalAcc = model.accessors[primitive.attributes.at("NORMAL")];
            normals = ReadAccessor<glm::vec3>(model, normalAcc);
        }
        else {
            normals.resize(positions.size(), glm::vec3(0));
        }

        std::vector<glm::vec2> texCoords;
        if (primitive.attributes.count("TEXCOORD_0"))
        {
            auto& texCoordAcc = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            texCoords = ReadAccessor<glm::vec2>(model, texCoordAcc);
        }
        else
        {
            texCoords.resize(positions.size(), glm::vec2(0));
        }

        vertices.reserve(vertices.size() + positions.size());
        for (size_t i = 0; i < positions.size(); i++)
        {
            vertices.emplace_back(Vertex{ positions[i], normals[i], texCoords[i] });
        }

        auto& idxAcc = model.accessors[primitive.indices];
        auto& view = model.bufferViews[idxAcc.bufferView];
        auto& buffer = model.buffers[view.buffer];

        auto dataPtr = buffer.data.data() + view.byteOffset + idxAcc.byteOffset;

        auto count = idxAcc.count;

        indices.reserve(indices.size() + count);

        switch (idxAcc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        {
            auto src = reinterpret_cast<const uint16_t*>(dataPtr);
            for (size_t i = 0; i < count; i++)
            {
                indices.emplace_back(src[i]);
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        {
            auto src = reinterpret_cast<const uint32_t*>(dataPtr);
            for (size_t i = 0; i < count; i++)
            {
                indices.emplace_back(src[i]);
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        {
            auto src = reinterpret_cast<const uint8_t*>(dataPtr);
            for (size_t i = 0; i < count; i++)
            {
                indices.emplace_back(src[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported index type");
        }

        return count;
    }

    void loadModel() {
        tinygltf::TinyGLTF loader{};

        tinygltf::Model model;
        std::string err;
        if (!loader.LoadASCIIFromFile(&model, &err, nullptr, "Models/CesiumMan.gltf")) {
            throw std::runtime_error("Failed to load model: " + err);
        }

        for (auto& mesh : model.meshes) {
            for (auto& primitive : mesh.primitives) {
                vk::DrawIndexedIndirectCommand drawCmd{
                    0, // index count
                    1, // instance count
                    static_cast<uint32_t>(indices.size()), // first index
                    static_cast<int32_t>(vertices.size()), // vertex offset
                    static_cast<uint32_t>(drawCmds.size()) // first instance
                };

                drawCmd.indexCount = LoadPrimitive(model, primitive);

                drawCmds.emplace_back(std::move(drawCmd));
            }
        }

        Texture texture{};

        int texChannels;
        auto pixels = stbi_load("Models/CesiumMan_img0.jpg", &texture.width, &texture.height, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        texture.imageData.resize(texture.width * texture.height * 4);
        memcpy(texture.imageData.data(), pixels, texture.imageData.size());

        stbi_image_free(pixels);

        textures.emplace_back(std::move(texture));

        Instance instance{};
        instance.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -0.5));
        instance.colour = glm::vec3(1.0f, 1.0f, 1.0f);

        instances.emplace_back(std::move(instance));
    }

    void createTextureResources() {
        textureImages.reserve(textures.size());

        vk::ImageCreateInfo imageInfo{};
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.format = vk::Format::eR8G8B8A8Srgb;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

        for (auto& texture : textures) {
            imageInfo.extent.width = texture.width;
            imageInfo.extent.height = texture.height;

            auto textureImage = rhi.createImage(imageInfo);

            rhi.updateImage(textureImage, texture.imageData);
            textureImages.emplace_back(std::move(textureImage));
        }

        vk::PhysicalDeviceProperties properties = rhi.getPhysicalDevice().getProperties();
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.anisotropyEnable = true;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

        textureSampler = vk::raii::Sampler(rhi.getDevice(), samplerInfo);
    }

    void createShadowResources() {
        shadowImages.reserve(rhi.getMaxFramesInFlight());

        auto extent = rhi.getSwapChainExtent();

        vk::ImageCreateInfo imageInfo{};
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.format = rhi.getDepthFormat();
        imageInfo.extent.width = extent.width;
        imageInfo.extent.height = extent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;

        for (size_t i = 0; i < rhi.getMaxFramesInFlight(); ++i) {
            shadowImages.emplace_back(std::move(rhi.createImage(imageInfo)));
        }

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

        shadowSampler = vk::raii::Sampler(rhi.getDevice(), samplerInfo);
    }

    void createVertexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;

        vertexBuffer = rhi.createBuffer(bufferInfo);
        rhi.updateBuffer(vertexBuffer, vertices);
    }

    void createIndexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(indices[0]) * indices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;

        indexBuffer = rhi.createBuffer(bufferInfo);
        rhi.updateBuffer(indexBuffer, indices);
    }

    void createIndirectBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(drawCmds[0]) * drawCmds.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst;

        indirectBuffer = rhi.createBuffer(bufferInfo);
        rhi.updateBuffer(indirectBuffer, drawCmds);
    }

    void createUniformBuffers() {
        uniformBuffers.reserve(rhi.getMaxFramesInFlight());

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(UniformBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

        for (size_t i = 0; i < rhi.getMaxFramesInFlight(); i++) {
            auto uniformBuffer = rhi.createBuffer(bufferInfo,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
            uniformBuffer.map();
            uniformBuffers.emplace_back(std::move(uniformBuffer));
        }
    }

    void createStorageBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(instances[0]) * instances.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;

        storageBuffer = rhi.createBuffer(bufferInfo);
        rhi.updateBuffer(storageBuffer, instances);
    }

    void createDescriptorPool() {
        auto maxFramesInFlight = rhi.getMaxFramesInFlight();

        std::array<vk::DescriptorPoolSize, 4> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, maxFramesInFlight },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(textures.size()) },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, maxFramesInFlight },
        };

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = maxFramesInFlight;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        descriptorPool = vk::raii::DescriptorPool(rhi.getDevice(), poolInfo);
    }

    void createDescriptorSets() {
        // storage buffer descriptor info (same buffer for all sets)
        vk::DescriptorBufferInfo storageBufferInfo{};
        storageBufferInfo.buffer = storageBuffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = instances.size() * sizeof(Instance);

        vk::DescriptorBufferInfo uboBufferInfo{};
        uboBufferInfo.offset = 0;
        uboBufferInfo.range = sizeof(UniformBufferObject);

        std::vector<vk::DescriptorImageInfo> imageInfos{};
        for (size_t i = 0; i < textures.size(); i++)
        {
            vk::DescriptorImageInfo imageInfo{};
            imageInfo.sampler = textureSampler;
            imageInfo.imageView = textureImages[i].getImageView();
            imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            imageInfos.emplace_back(std::move(imageInfo));
        }

        vk::WriteDescriptorSet uboWrite{};
        uboWrite.dstBinding = 0;
        uboWrite.dstArrayElement = 0;
        uboWrite.descriptorCount = 1;
        uboWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        uboWrite.pBufferInfo = &uboBufferInfo;

        vk::WriteDescriptorSet ssboWrite{};
        ssboWrite.dstBinding = 1;
        ssboWrite.dstArrayElement = 0;
        ssboWrite.descriptorCount = 1;
        ssboWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
        ssboWrite.pBufferInfo = &storageBufferInfo;

        vk::WriteDescriptorSet imageWrite{};
        imageWrite.dstBinding = 2;
        imageWrite.dstArrayElement = 0;
        imageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        imageWrite.descriptorCount = imageInfos.size();
        imageWrite.pImageInfo = imageInfos.data();

        vk::DescriptorImageInfo shadowImageInfo{};
        shadowImageInfo.sampler = shadowSampler;
        shadowImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        vk::WriteDescriptorSet shadowImageWrite{};
        shadowImageWrite.dstBinding = 3;
        shadowImageWrite.dstArrayElement = 0;
        shadowImageWrite.descriptorCount = 1;
        shadowImageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;

        std::vector<vk::DescriptorSetLayout> layouts(rhi.getMaxFramesInFlight(), mainPipeline.getDescriptorSetLayout());

        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data(); 

        descriptorSets = rhi.getDevice().allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < rhi.getMaxFramesInFlight(); i++) {
            uboBufferInfo.buffer = uniformBuffers[i];
            uboWrite.dstSet = descriptorSets[i];
            ssboWrite.dstSet = descriptorSets[i];
            imageWrite.dstSet = descriptorSets[i];

            shadowImageInfo.imageView = shadowImages[i].getImageView();

            shadowImageWrite.dstSet = descriptorSets[i];
            shadowImageWrite.pImageInfo = &shadowImageInfo;

            rhi.getDevice().updateDescriptorSets(uboWrite, {});
            rhi.getDevice().updateDescriptorSets(ssboWrite, {});
            rhi.getDevice().updateDescriptorSets(imageWrite, {});
            rhi.getDevice().updateDescriptorSets(shadowImageWrite, {});
        }
    }

    void initRenderGraph()
    {
        // Shadow pass: render scene from light into depth buffer
        Gfx::RenderPassNode shadowPass{ "ShadowPass" };

        Gfx::RenderPassNode::AttachmentTransitionInfo shadowTransition{ {}, vk::ImageAspectFlagBits::eDepth };
        shadowTransition.images.resize(shadowImages.size()); // populate with the vk::Image handles for each per-frame shadow image
        for (size_t i = 0; i < shadowImages.size(); ++i) shadowTransition.images[i] = *shadowImages[i];
        shadowTransition.oldLayout = vk::ImageLayout::eUndefined;
        shadowTransition.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        shadowTransition.srcAccessMask = vk::AccessFlagBits2::eNone;
        shadowTransition.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        shadowTransition.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        shadowTransition.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        shadowPass.transitionInfos.emplace_back(shadowTransition);

        shadowPass.recordFunc = [this](vk::raii::CommandBuffer& cmd, uint32_t imageIndex)
            {
                auto swapChainExtent = rhi.getSwapChainExtent();

                updateUniformBuffer(imageIndex);

                cmd.bindVertexBuffers(0, *vertexBuffer, { 0 });
                cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);

                cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
                cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

                vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0.0f);
                vk::RenderingAttachmentInfo shadowAttachmentInfo{};
                shadowAttachmentInfo.imageView = shadowImages[imageIndex].getImageView();
                shadowAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
                shadowAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
                shadowAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
                shadowAttachmentInfo.clearValue = clearDepth;

                vk::RenderingInfo renderingInfo{};
                renderingInfo.renderArea.offset.x = 0;
                renderingInfo.renderArea.offset.y = 0;
                renderingInfo.renderArea.extent = swapChainExtent;
                renderingInfo.layerCount = 1;
                renderingInfo.pDepthAttachment = &shadowAttachmentInfo;

                cmd.beginRendering(renderingInfo);

                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, shadowPipeline);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, shadowPipeline.getPipelineLayout(), 0, *descriptorSets[imageIndex], nullptr);
                cmd.drawIndexedIndirect(*indirectBuffer, 0, drawCmds.size(), static_cast<uint32_t>(sizeof(VkDrawIndexedIndirectCommand)));

                cmd.endRendering();
            };

        graph.addPass(shadowPass);

        // Main pass: render scene from camera, sampling shadow map
        Gfx::RenderPassNode mainPass{ "MainPass" };

        // Transition shadow image from depth attachment -> shader read for sampling in main pass
        shadowTransition.oldLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        shadowTransition.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        shadowTransition.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        shadowTransition.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        shadowTransition.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests;
        shadowTransition.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        mainPass.transitionInfos.emplace_back(std::move(shadowTransition));

        Gfx::RenderPassNode::AttachmentTransitionInfo mainColorTransition{ rhi.getSwapChain().getImages(), vk::ImageAspectFlagBits::eColor };
        mainColorTransition.oldLayout = vk::ImageLayout::eUndefined;
        mainColorTransition.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        mainColorTransition.srcAccessMask = vk::AccessFlagBits2::eNone;
        mainColorTransition.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        mainColorTransition.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        mainColorTransition.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        mainPass.transitionInfos.emplace_back(mainColorTransition);

        Gfx::RenderPassNode::AttachmentTransitionInfo mainDepthTransition{ rhi.getDepthImages(), vk::ImageAspectFlagBits::eDepth };
        mainDepthTransition.oldLayout = vk::ImageLayout::eUndefined;
        mainDepthTransition.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        mainDepthTransition.srcAccessMask = vk::AccessFlagBits2::eNone;
        mainDepthTransition.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        mainDepthTransition.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        mainDepthTransition.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        mainPass.transitionInfos.emplace_back(std::move(mainDepthTransition));

        mainPass.recordFunc = [this](vk::raii::CommandBuffer& cmd, uint32_t imageIndex)
            {
                auto swapChainExtent = rhi.getSwapChainExtent();

                vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
                vk::RenderingAttachmentInfo colorAttachmentInfo{};
                colorAttachmentInfo.imageView = rhi.getSwapChainImageView(imageIndex);
                colorAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
                colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
                colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
                colorAttachmentInfo.clearValue = clearColor;

                vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1, 0);
                vk::RenderingAttachmentInfo depthAttachmentInfo{};
                depthAttachmentInfo.imageView = rhi.getDepthImageView(imageIndex);
                depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
                depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
                depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
                depthAttachmentInfo.clearValue = clearDepth;

                vk::RenderingInfo renderingInfo{};
                renderingInfo.renderArea.offset.x = 0;
                renderingInfo.renderArea.offset.y = 0;
                renderingInfo.renderArea.extent = swapChainExtent;
                renderingInfo.layerCount = 1;
                renderingInfo.colorAttachmentCount = 1;
                renderingInfo.pColorAttachments = &colorAttachmentInfo;
                renderingInfo.pDepthAttachment = &depthAttachmentInfo;

                cmd.beginRendering(renderingInfo);

                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, mainPipeline);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, mainPipeline.getPipelineLayout(), 0, *descriptorSets[imageIndex], nullptr);
                cmd.drawIndexedIndirect(*indirectBuffer, 0, drawCmds.size(), static_cast<uint32_t>(sizeof(VkDrawIndexedIndirectCommand)));

                cmd.endRendering();
            };

        graph.addPass(mainPass);

        // Present transition pass: transition main color image from color attachment -> present for presentation to swap chain
        Gfx::RenderPassNode presentTransition{ "PresentTransition" };
        mainColorTransition.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        mainColorTransition.newLayout = vk::ImageLayout::ePresentSrcKHR;
        mainColorTransition.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        mainColorTransition.dstAccessMask = vk::AccessFlagBits2::eNone;
        mainColorTransition.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        mainColorTransition.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        presentTransition.transitionInfos.emplace_back(std::move(mainColorTransition));

        graph.addPass(presentTransition);

        graph.init();
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        auto swapChainExtent = rhi.getSwapChainExtent();

        UniformBufferObject ubo{};
        ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        ubo.rotation = glm::angleAxis(time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        auto nLightDir = -glm::normalize(glm::vec3(-1.0f, 1.0, -1.0));
        ubo.nLightDir = glm::vec4(nLightDir, 0.0f);
        ubo.lightView = lookAt(nLightDir, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.lightProj = glm::ortho(-3.0f, 3.0f, -3.0f, 3.0f, 0.1f, 10.0f);
        ubo.lightProj[1][1] *= -1;

        memcpy(uniformBuffers[currentImage].getMappedData(), &ubo, sizeof(ubo));
    }

    void drawFrame() {
        graph.executeFrame();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        rhi.getDevice().waitIdle();
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

        std::cerr << e.what() << std::endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
