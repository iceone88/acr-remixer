-- StageTest (acr-stagegen): camera tour of a generated stage for screenshots.
--  * once the player is in a car (possessed pawn is not a camera) and ~10 s
--    have passed, switch to the debug camera and fly its SpectatorPawn to each
--    point in ../points.txt ("label x y z yaw", Unreal cm / degrees)
--  * logs "[StageTest] at <label> want=(..) cam=(..)" when in place, "done" at the end
--  * live hook: ../live.lua is re-run whenever it changes (iterate without restarts)
-- Teleporting the physics car itself crashes the game; only the camera moves.
local here = debug.getinfo(1, "S").source:sub(2):gsub("\\", "/"):match("(.*/)")
local POINTS, LIVE = here .. "../points.txt", here .. "../live.lua"

local function v(l) return l and string.format("(%.0f, %.0f, %.0f)", l.X / 100, l.Y / 100, l.Z / 100) or "nil" end

local function load_points()
    local f = io.open(POINTS, "r")
    if not f then return nil end
    local t = {}
    for line in f:lines() do
        local lab, x, y, z, yaw = line:match("^(%S+)%s+(%S+)%s+(%S+)%s+(%S+)%s+(%S+)")
        if lab then table.insert(t, { lab, tonumber(x), tonumber(y), tonumber(z), tonumber(yaw) }) end
    end
    f:close()
    return t
end

local function controller()
    for _, pc in ipairs(FindAllOf("PlayerController") or {}) do
        if pc:IsValid() and pc:GetClass():GetFName():ToString() ~= "DebugCameraController" then return pc end
    end
end

local function console(cmd)
    local pc = controller()
    if not pc then return false end
    StaticFindObject("/Script/Engine.Default__KismetSystemLibrary"):ExecuteConsoleCommand(pc, cmd, pc)
    return true
end

local function in_car()
    local pc = controller()
    local p = pc and pc.Pawn
    if not (p and p:IsValid()) then return false end
    return not string.find(p:GetClass():GetFName():ToString(), "Camera")
end

local function debug_cam(on)
    local dcc = FindFirstOf("DebugCameraController")
    local active = (dcc ~= nil) and dcc:IsValid()
    if on ~= active then console("ToggleDebugCamera") end
end

local function view_at(pt)
    debug_cam(true)
    local loc = { X = pt[2], Y = pt[3], Z = pt[4] + 400 }
    local rot = { Pitch = -12, Yaw = pt[5], Roll = 0 }
    local sp = FindFirstOf("SpectatorPawn")
    if sp and sp:IsValid() then sp:K2_TeleportTo(loc, rot) end
    local dcc = FindFirstOf("DebugCameraController")
    if dcc and dcc:IsValid() then dcc:SetControlRotation(rot) end
end

local function cam_loc()
    local ok, r = pcall(function() return v(FindFirstOf("DebugCameraController").PlayerCameraManager:GetCameraLocation()) end)
    return ok and r or "?"
end

-- tour state machine, ticks every 0.5 s
local state, points, i, t, done = "wait_car", nil, 0, 0, false
local function restart() state, points, i, t, done = "wait_car", nil, 0, 0, false end
StageTestAPI = { console = console, view_at = view_at, restart = restart, debug_cam = debug_cam }

LoopAsync(500, function()
    local ok, err = pcall(function()
        if done then return end
        t = t + 1
        if state == "wait_car" then
            if in_car() then
                points = load_points()
                if points and #points > 0 then
                    state, t = "settle", 0
                    print(string.format("[StageTest] in car, %d points\n", #points))
                end
            else
                t = 0
            end
        elseif state == "settle" and t >= 20 then        -- ~10 s for the stage to stream in
            state, i, t = "move", 0, 0
        elseif state == "move" then
            i = i + 1
            if i > #points then
                debug_cam(false)
                print("[StageTest] done\n")
                done = true
                return
            end
            view_at(points[i])
            state, t = "hold", 0
        elseif state == "hold" and t >= 5 then             -- 2.5 s to stream, then shoot
            print(string.format("[StageTest] at %s want=(%.0f, %.0f) cam=%s\n", points[i][1],
                points[i][2] / 100, points[i][3] / 100, cam_loc()))
            state, t = "shot", 0
        elseif state == "shot" and t >= 2 then             -- host grabs within 1 s
            state = "move"
        end
    end)
    if not ok then print("[StageTest] error: " .. tostring(err) .. "\n") end
    return false
end)

local live_src = nil
LoopAsync(1000, function()
    local f = io.open(LIVE, "r")
    if f then
        local src = f:read("*a"); f:close()
        if live_src == nil then live_src = src end           -- don't re-run a stale file at startup
        if src ~= live_src then
            live_src = src
            local chunk, e = load(src, "live")
            local ok, r = false, e
            if chunk then ok, r = pcall(chunk) end
            print("[StageTest] live: " .. (ok and "ok" or tostring(r)) .. "\n")
        end
    end
    return false
end)
print("[StageTest] loaded\n")
