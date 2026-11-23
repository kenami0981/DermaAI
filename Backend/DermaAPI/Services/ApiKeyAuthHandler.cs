using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Options;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;

namespace Derma.API.Services
{
    public class ApiKeyAuthHandler : AuthenticationHandler<AuthenticationSchemeOptions>
    {
        private readonly IConfiguration _config;

        public ApiKeyAuthHandler(
            IOptionsMonitor<AuthenticationSchemeOptions> options,
            ILoggerFactory logger,
            UrlEncoder encoder,
            IConfiguration config) : base(options, logger, encoder)
        {
            _config = config;
        }

        protected override Task<AuthenticateResult> HandleAuthenticateAsync()
        {
            //API key musi być traktowany jako admin
            var identity = new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.Name, "AI"),
                new Claim(ClaimTypes.Role, "Admin")
            }, Scheme.Name);
            if (!Request.Headers.TryGetValue("X-Api-Key", out var provided))
                return Task.FromResult(AuthenticateResult.NoResult());

            var expected = _config["ApplicationSettings:AI_ApiKey"];

            if (string.IsNullOrEmpty(expected) ||
                !CryptographicOperations.FixedTimeEquals(
                    Encoding.UTF8.GetBytes(provided.ToString()),
                    Encoding.UTF8.GetBytes(expected)))
                return Task.FromResult(AuthenticateResult.Fail("Nieprawidłowy klucz API."));


            return Task.FromResult(AuthenticateResult.Success(
                new AuthenticationTicket(new ClaimsPrincipal(identity), Scheme.Name)));
        }
    }
}